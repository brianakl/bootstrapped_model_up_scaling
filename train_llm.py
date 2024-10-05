import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import bus_decoder_model
from torch.utils.tensorboard.writer import SummaryWriter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    data = None
    if dataset == 'shakespeare':
        data = load_dataset('tiny_shakespeare')

