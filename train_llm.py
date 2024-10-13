import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from bus_decoder_model import Decoder, Transformer, AttentionHead
from torch.utils.tensorboard.writer import SummaryWriter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Params
d_model = 368 
target_size = 512
num_layers = 4
num_heads = 8
d_hidden = 4*d_model
seq_len = 128
vocab_size = 67
max_steps = 5000
transfer_step = 800
dataset = 'salesforce/wikitext'
lr = 1e-3
min_lr = 1e-4
batch_size = 128

def get_data(dataset='salesforce/wikitext'):
    data = load_dataset(dataset)
    # TODO: process and clean data

def test_model(model, data):
    pass

def train_model(model, data, transfer_step=900, target_size=1024, lr=1e-3, min_lr=1e-6, max_iters=5000, transfer=False):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, min_lr)
    writer = SummaryWriter()
    for iter in tqdm.tqdm(range(1, max_iters)):

        if iter == transfer_step and transfer:
        # if iter <= 1000 and iter % 500 == 0:
            model.expand(target_size)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
            print('at step {}: expanded model to: {} M parameters'.format(iter, sum(p.numel() for p in model.parameters())/1e6))
            model.to('cpu')
            model.to(DEVICE)    # Shortcut to recompile gradient backprop since the model changed sizes (assuming always using gpu)
            loss_func = torch.nn.CrossEntropyLoss()

        # evaluate the loss
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        loss = loss_func(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        writer.add_scalar('Loss/train', loss, iter)
        optimizer.step()
        scheduler.step()

    writer.flush()
    writer.close()
    test_model(model, data['test'])










if __name__ == '__main__':
    # TODO: arg parsing
    parser = argparse.ArgumentParser(description="LLM training args")
    # Add a required positional argument called "filename"
    parser.add_argument("filename", help="The name of the file to process")
    
    # Add an optional boolean flag called "verbose"
    parser.add_argument("-v", "--verbose", action="store_true", help="Print extra information about the processing")
    parser.add_argument("--d_model", action="store_true", default=d_model, help="model dimensions")
    parser.add_argument("--target_size", action="store_true", default=target_size, help="model dimensions")
    parser.add_argument("--dataset", action="store_true", default=dataset, help="Dataset")
    parser.add_argument("--num_layers", action="store_true", default=num_layers, help="Dataset")
    parser.add_argument("--d_hidden", action="store_true", default=d_hidden, help="Dataset")
    parser.add_argument("--vocab_size", action="store_true", default=vocab_size, help="Dataset")
    parser.add_argument("--num_heads", action="store_true", default=num_heads, help="Dataset")
    parser.add_argument("--seq_len", action="store_true", default=seq_len, help="Dataset")
    parser.add_argument("--lr", action="store_true", default=lr, help="Dataset")
    parser.add_argument("--min_lr", action="store_true", default=min_lr, help="Dataset")
    parser.add_argument("--steps", action="store_true", default=max_steps, help="Dataset")
    parser.add_argument("--transfer", action="store_true", default=transfer_step, help="Dataset")
    parser.add_argument("--batch_size", action="store_true", default=batch_size, help="Dataset")

    print("Device: ", DEVICE)

    data = None
    if dataset == 'shakespeare':
        data = load_dataset('tiny_shakespeare')

    model = Decoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_hidden=d_hidden, vocab_size=vocab_size, seq_len=seq_len)

    train_model(model=model, data=data, transfer_step=transfer_step)

