import time 
import torch 
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
import tqdm


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()

        self.W_Q = torch.nn.Linear(d_model, d_internal, False)
        self.W_K = torch.nn.Linear(d_model, d_internal, False)
        self.W_V = torch.nn.Linear(d_model, d_internal, False)

        self.SoftMax = torch.nn.Softmax(dim=-1)


        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_internal),
            torch.nn.ReLU(), 
            torch.Dropout(),
            torch.nn.Linear(d_internal, d_model)
        )
        self.d_model = d_model
        self.d_internal = d_internal

        self.double()


    def forward(self, input_vecs):

        Q = self.W_Q(input_vecs)
        K = self.W_K(input_vecs)
        V = self.W_V(input_vecs)

        Q = torch.matmul(Q, torch.transpose(K, -2, -1))
        Q = Q / torch.sqrt(torch.tensor(self.d_model))

        Attn = self.SoftMax(Q)
        a = torch.matmul(Attn, V)
        a += input_vecs

        output = self.FFN(a) + a

        return output, Attn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, num_positions: int=20, batched=False):
        super().__init__()

        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched
        self.d_model = d_model
        self.num_positions = num_positions
        
        self.sinu = torch.zeros((num_positions, d_model))

        for pos in range(num_positions):
            for m in range(d_model):
                if m%2 == 0:
                    self.sinu[pos][m] += torch.sin(torch.tensor(pos/(10000**((2*m)/self.d_model))))
                else:
                    self.sinu[pos][m] += torch.cos(torch.tensor(pos/(10000**((2*m)/self.d_model))))


    def forward(self, x):
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)

        if self.batched:
            for b in range(x.shape[0]):
                x[b] += self.sinu
            return x

        else:
            return x + self.sinu






class LetterCountingExample(object):
    def __init__(self, input:str, output:np.array, vocab_index:Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)



class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, **args):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_internal = d_internal

        self.tformer = TransformerLayer(d_model, d_internal)
        self.Softmax = torch.nn.LogSoftmax(dim=-1)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_internal),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(d_internal, num_classes)
        )
        self.b = False
        self.penc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=self.b)
        self.embed = torch.nn.Embedding(vocab_size, d_model)

        self.double()



    def extrap(self, model, method='double'):
        if method == 'double':
            d_m = self.d_model
            d_i = d_m // 2
            z = torch.zeros(d_m, d_i, dtype=float)

            j = 0
            for i in range(0, d_m-1, 2):
                z[i][j] = 1
                z[i+1][j] = 1
                j += 1

            z1 = torch.zeros(d_i, d_i//2, dtype=float)
            j = 0
            for i in range(0, d_i-1, 2):
                z1[i][j] = 1
                z1[i+1][j] = 1
                j += 1
            

            # TODO: check that all this matmul is correct
            Q = torch.matmul(z, torch.transpose(model.tformer.W_Q.weight.data, -1, -2))
            Q = torch.matmul(Q, torch.transpose(z1, -1, -2))
            self.tformer.W_Q.weight.data = torch.transpose(Q, -1, -2)

            K = torch.matmul(z, torch.transpose(model.tformer.W_K.weight.data, -1, -2))
            K = torch.matmul(z1, torch.transpose(K, -1, -2))
            self.tformer.W_K.weight.data = K

            V = torch.matmul(z, torch.transpose(model.tformer.W_V.weight.data, -1, -2))     # especially this one for V
            V = torch.matmul(z, torch.transpose(z1, -1, -2))
            self.tformer.W_V.weight.data = torch.transpose(V, -1, -2)


    def forward(self, indices):
        t = self.embed(indices)
        t = self.penc(t)
        t = t.to(torch.float64)
        t, attn = self.tformer(t)
        x = self.FFN(t)
        x = self.Softmax(x)
        return x, [attn]


    def batch(self, b):
        self.b = b
        self.penc.batched = b














