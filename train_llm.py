import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
from bus_nGPT import *
from torch.utils.tensorboard.writer import SummaryWriter
from collections import defaultdict, Counter
# import wandb
import json
import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Params
d_model = 256 
target_size = 512
num_layers = 8
num_heads = 8
d_hidden = 4*d_model
con_len = 512
vocab_size = 30523
total_steps = 5000 
transfer_step = 800
dataset = 'openwebtext'
lr = 1e-3
min_lr = 1e-4
batch_size = 10
grad_accumulation_steps=20
eval_interval=100
eval_samples=400
name='test'
num_transfer_steps=800
wb = True
model_saving = False


# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split='train', batch_size=128, con_len=128):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - con_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+con_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+con_len]).astype(np.int64)) for i in ix])
    if DEVICE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def test_model(model, data):
    pass

def save_model(model, name, steps):
    torch.save(model, name+steps+".pt")

def train(model:Decoder, 
          lr=1e-3, 
          grad_accum_steps=100, 
          eval_interval=100,
          min_lr=1e-4,
          name='std_model',
          batch_size=128,
          transfer_step_size=64,
          num_transfer_steps=1,
          transfer_step=2000,
          eval_samples=1000,
          total_steps=20000,
          model_saving=False,
          con_len=128
          ):

    
    writer = SummaryWriter(comment=name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # breakpoint()
    model = torch.compile(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, min_lr)

    scaler = torch.GradScaler()

    last_transfer_step = transfer_step * num_transfer_steps


    expand = False
    
    steps = 1
    last_bus = 1
    transfer_steps_count = 0
    iter = 1

    t = tqdm.tqdm(total=total_steps, desc='Steps')

    avg_loss = torch.tensor(0., device=DEVICE)

    # breakpoint()
    optimizer.zero_grad(set_to_none=True)

    while steps <= total_steps:
        
        if transfer_step != -1 and (steps%transfer_step == 0 and last_bus != steps) and steps <= last_transfer_step:
            avg_loss = torch.tensor(0., device=DEVICE)
            target_size = model.d_model + transfer_step_size
            # target_size = model.d_model * 2
            print('model expanded: ', target_size)
            batch_size = int(batch_size // (target_size/model.d_model))

            grad_accum_steps = round(102400/(batch_size*con_len))
            print("bsz: ", batch_size)
            print('grad steps: ', grad_accum_steps)
            model.expand(target_size)
            print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, min_lr)
            expand = 1
            model.to('cpu')
            model.to(DEVICE)    # used to recompile compute graph to reflect new model
            last_bus = steps
            transfer_steps_count += 1

        if ((iter%(eval_interval*grad_accum_steps) == 0) or expand or iter == 1):
            l = 0
            print("running eval")
            with torch.no_grad():
                model.eval()
                for _ in range(eval_samples):
                    x, y = get_batch('val', batch_size=batch_size, con_len=con_len)
                    py = model(x)
                    B, T, C = py.shape
                    logits = py.view(B*T, C)
                    targets = y.view(B*T)
                    l += F.cross_entropy(logits, targets)
                total_l = l/eval_samples
                writer.add_scalar("Val loss", total_l, steps)
                print("val loss: ", total_l)
                expand = 0
                # bus = False if last_val * 0.9 >= total_l else True
                # last_val = total_l
                if model_saving: save_model(model, name, steps)
            model.train()

        xb, yb = get_batch(split='train', batch_size=batch_size, con_len=con_len)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1)
        scaler.scale(loss).backward()
        # loss.backward()
        avg_loss += loss

        
        if iter % grad_accum_steps == 0: 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            x = 0

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            l = avg_loss.cpu().item()/grad_accum_steps
            writer.add_scalar("Training Perplexity", np.exp(l), steps)
            writer.add_scalar("Training Loss", l, steps)
            t.update()
            scheduler.step()
            model.normalize()
            avg_loss = torch.tensor(0., device=DEVICE)
            steps += 1

        iter += 1

    t.close()
    writer.close()




if __name__ == '__main__':

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    bos_token = "<|BOS|>"
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer.add_special_tokens({"bos_token": bos_token})
    vocab_size = tokenizer.vocab_size


    d_model = config['d_model']
    target_size = config['target_size']
    num_layers = config['num_layers']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    con_len = config['con_len']
    lr = config['lr']
    min_lr = config['min_lr']
    steps = config['steps']
    transfer = config['transfer']
    batch_size = config['batch_size']
    name = config['name']
    grad_accumulation_steps = config['grad_accumulation_steps']
    eval_samples = config['eval_samples']
    eval_interval = config['eval_interval']
    model_saving = config['model_saving']

    print("Model Config")
    for key in config.keys():
        print(f"{key+':':<30}{config[key]}")

    model = Decoder(d_model=d_model, 
                    num_layers=num_layers, 
                    num_heads=num_heads, 
                    vocab_size=vocab_size, 
                    )

    print("Device: ", DEVICE)
    print("{}M Parameters".format(sum([p.numel() for p in model.parameters()])/1e6))
    model.to(DEVICE)

    train(model=model, 
          transfer_step=transfer_step,
          min_lr=min_lr,
          grad_accum_steps=grad_accumulation_steps,
          lr=lr,
          name=name,
          num_transfer_steps=num_transfer_steps,
          total_steps=total_steps,
          con_len=con_len,
          eval_interval=eval_interval,
          eval_samples=eval_samples,
          model_saving=model_saving,
         )
    save_model(model, name+'_final', '')
 
