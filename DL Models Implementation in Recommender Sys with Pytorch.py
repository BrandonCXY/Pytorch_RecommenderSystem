#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
    DeepLearning Models Implementation in Recommender Sys 
    So far including: LR, FM, Wide&Deep, DepFM, NFM
    From: https://github.com/rixwew/pytorch-fm/tree/acc6997eeabefe2cd6222b1e684306c8b4fcc8e9
'''


# In[26]:


import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import lmdb
import torch.utils.data


# In[6]:


class MovieLens1MDataset(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep='::', engine='python', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because in original dataset the ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1 # field_dims = array([6040, 3952])
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


# In[33]:


class CriteoDataset(torch.utils.data.Dataset):
    """
    Criteo Display Advertising Challenge Dataset
    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat 
        them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of 
        Criteo Competition
    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.
    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='/home/dm/Downloads/Kaggle_Criteo/Cache',                  rebuild_cache=True, min_threshold=10):
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        # here path is the dataset path
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm.tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold}                        for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        '''
        feat_mapper: {第几个feature:{feature value:idx}}
        defaults: {第几个feature:该feature个数}
        '''
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm.tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)


# In[8]:


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            #torch.save(model, self.save_path)
            '''
            Save:
            torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
            
            Load:
            model = describe_model()
            checkpoint = torch.load('checkpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            '''
            torch.save({'state_dict': model.state_dict()}, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


# In[9]:


'''
    Layers
'''

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        #sum(field_dims) 是为了把所有的field结合在一起
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim) #相当于是FM里的matrix V(n*k)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # 这里为什么取[:-1]? 这样0-9992 的embedding表中，0-6040是对应user，6041-9992对应item
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long) #array([   0, 6040]) 
        
        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)`` num_fields = 2 当只有userid 和itemid时
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0) #只是简单地加上6040
        return torch.sum(self.fc(x), dim=1) + self.bias # output size (batch_size, num_fields)


class FeaturesEmbedding(torch.nn.Module):
    '''
    通过FeaturesEmbedding之后，相当于直接通过 X · V 进行筛选，只有xi=1相对应的vi才会放入output  
    '''
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        #sum(field_dims) 是为了把所有的field结合在一起
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # 这里为什么取[:-1]? 这样0-9992 的embedding表中，0-6040是对应user，6041-9992对应item
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x) # output size (batch_size, num_fields, embed_dim)  
    

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
        # if reduce_sum = False, ouput size is (batch_size, embed_dim)
        # if reduce_sum = True, ouput size is (batch_size, 1)
    

class MultiLayerPerceptron(torch.nn.Module):
                                
    def __init__(self, input_dim, hidden_layers, dropout, output_layer=True): # input_dim = 64, embed_dim = (64,)
        super().__init__()
        layers = list()
        for hidden_layer_size in hidden_layers: # for i in (64,) print i ---> 64
            layers.append(torch.nn.Linear(input_dim, hidden_layer_size))
            # generally put batchnorm after linearization and before activation
            layers.append(torch.nn.BatchNorm1d(hidden_layer_size)) 
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_layer_size
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


# In[91]:


# test
dataset = MovieLens1MDataset('/home/dm/Downloads/movielens1m/ratings.dat')
loader = DataLoader(dataset, batch_size=56, num_workers=8)
for i ,(a,b) in enumerate(loader):
    x = a 
    break
#x.shape
#x
embedding = FeaturesEmbedding(dataset.field_dims,10)
e = embedding(x)
#e.shape
#e


# In[10]:


class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """
    def _init_(self,field_dim):
        super()._init_()
        self.linear = FeaturesLinear(field_dim)
    def forward(self,x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))


# In[11]:


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


# In[12]:


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


# In[13]:


class FactorizationSupportedNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # 不应该是需要pre-training？为什么FNN和wide&deep的deep部分一模一样？
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


# In[14]:


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """
    #return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1]) #这里embed_dim 就是MLP的input_dim

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #embedding(x) ouptut size is (batchsize, num_field, embed_dim)
        cross_term = self.fm(self.embedding(x)) 
        #print(cross_term.shape)
        # cross_term size is (batch_size, embed_size)
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


# In[15]:


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        # tensor中的数据按照行优先的顺序排成一个一维的数据,所以可以按顺序完美排成 (batch_size，len(field_dims) * embed_dim）
        # DeepFM中，“we include the FM model as a part of overall learning architecture”
        # fm用的embedding(x)和mlp用的embedding(x)是一个，所以可以同时一起更新
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


# In[16]:


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
    #for i, (fields, target) in enumerate(data_loader):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval) #设置进度条右边显示的信息
            total_loss = 0


# In[17]:


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


# In[18]:


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


# In[19]:


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)


# In[20]:


def main(dataset_name,
         dataset_path,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         model_name,
         save_dir):
    device = torch.device(device)
    #dataset = MovieLens1MDataset('/home/dm/Downloads/movielens1m/ratings.dat')
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pth.tar')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


# In[34]:


dataset_name = 'criteo' 
dataset_path = '/home/dm/Downloads/Kaggle_Criteo/train.txt' 
#dataset_name = 'movielens1M'
#dataset_path = '/home/dm/Downloads/movielens1m/ratings.dat'


epoch = 20
learning_rate = 0.001
batch_size = 256
weight_decay = 1e-5
device = 'cuda:0'
model_name = 'dfm'
save_dir = '/home/dm/Downloads/models'
main(dataset_name,
         dataset_path,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         model_name,
         save_dir)


# In[ ]:




