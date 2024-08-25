import time 
import torch 
import torch.nn as nn
import numpy as np
import numpy.linalg as LA
import random
from torch import full, optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
import tqdm
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind


from transfer import TransformerLayer, PositionalEncoding, Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=3, d_model=512):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        





if __name__ == "__main__":
    model_args = [
        # {'vocab_size':27, 'num_positions':20, 'd_model':24, 'd_internal':12, 'num_classes':3, 'num_layers':1},
        # {'vocab_size':27, 'num_positions':20, 'd_model':48, 'd_internal':24, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':96, 'd_internal':48, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':192, 'd_internal':96, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':384, 'd_internal':192, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':768, 'd_internal':384, 'num_classes':3, 'num_layers':1},
    ]
    print("CUDA Device used: ", DEVICE)
    compare(model_args)
    # eigen_comparison(model_args)



































