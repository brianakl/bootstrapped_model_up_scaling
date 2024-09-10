import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_internal, vocab_size, num_heads):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.SoftMax = torch.nn.LogSoftmax(dim=-1)
        self.blocks = [Transformer(d_model, d_internal, vocab_size) for _ in range(num_heads)]

        self.connection = torch.nn.Linear(d_model, vocab_size//2),
        self.FFN = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(vocab_size//2, vocab_size),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        t = x
        for head in self.heads:
            t = head(t) + x

        t = self.connection(t)
        ret = self.FFN(t)

        return self.SoftMax(ret)

    def expand(self, d_mnew, d_inew):
        self.connection = torch.nn.Linear(d_mnew, self.vocab_size//2)
        for block in self.blocks:
            block.expand(d_mnew, d_inew)

        self.d_model = d_mnew
        self.d_internal = d_inew


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()

        self.W_Q = torch.nn.Linear(d_model, d_internal, False)
        self.W_K = torch.nn.Linear(d_model, d_internal, False)
        self.W_V = torch.nn.Linear(d_model, d_model, False)

        self.SoftMax = torch.nn.Softmax(dim=-1)


        # self.FFN = torch.nn.Sequential(
        #     torch.nn.Linear(d_model, d_model),
        #     torch.nn.ReLU(), 
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(d_model, d_model)
        # )
        self.d_model = d_model
        self.d_internal = d_internal

        self.double()

    def expand(self, d_mnew, d_inew):
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_mnew, d_mnew),
            torch.nn.ReLU(), 
            torch.nn.Dropout(),
            torch.nn.Linear(d_mnew, d_mnew)
        )
        self.W_Q.weight.data = torch.cat([self.W_Q.weight.data, torch.zeros(d_mnew-self.d_model, self.d_internal)], dim=0)
        self.w_q.weight.data = torch.cat([self.W_Q.weight.data, torch.zeros(d_mnew, d_inew - self.d_internal)], dim=1)
        for i in range(self.d_internal, d_inew):
            self.W_Q.weight.data[i][i] = self.W_Q.weight.data[i][i] if self.W_Q.weight.data[i][i] != 0 else 1

        self.W_K.weight.data = torch.cat([self.W_K.weight.data, torch.zeros(d_mnew - self.d_model, self.d_internal)], dim=0)
        self.W_K.weight.data = torch.cat([self.W_K.weight.data, torch.zeros(d_mnew, d_inew - self.d_internal )], dim=1)
        for i in range(self.d_internal, d_inew):
            self.W_K.weight.data[i][i] = self.W_K.weight.data[i][i] if self.W_K.weight.data[i][i] != 0 else 1

        self.W_V.weight.data = torch.cat([self.W_V.weight.data, torch.zeros(d_mnew - self.d_model, self.d_model)], dim=0)
        self.W_V.weight.data = torch.cat([self.W_V.weight.data, torch.zeros(d_mnew, d_mnew)], dim=1)
        for i in range(self.d_model, d_mnew):
            self.W_V.weight.data[i][i] = self.W_V.weight.data[i][i] if self.W_V.weight.data[i][i] != 0 else 1

        self.d_internal = d_inew
        self.d_model = d_mnew 


    def forward(self, input_vecs):
        Q = self.W_Q(input_vecs)
        K = self.W_K(input_vecs)
        V = self.W_V(input_vecs)

        Q = torch.matmul(Q, torch.transpose(K, -2, -1))
        Q = Q / torch.sqrt(torch.tensor(self.d_model))
        

        Attn = self.SoftMax(Q)
        a = torch.matmul(Attn, V)

        return a

class Transformer(nn.Module):
    def __init__(self, d_model, vocab_size, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_internal = int(d_model/num_heads)
        self.num_heads = num_heads
        self.vocab_size = num_heads

        self.heads= [AttentionHead(d_model, self.d_internal) for _ in range(num_heads)]
        self.Softmax = torch.nn.LogSoftmax(dim=-1)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(num_heads*d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(d_model, vocab_size)
        )
        self.W_O = torch.nn.Linear(d_model, d_model, False)
        self.b = False
        self.layernorm = torch.nn.LayerNorm(d_model)
        # self.penc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=self.b)
        # self.embed = torch.nn.Embedding(vocab_size, d_model).to(DEVICE)

        self.double()


    def forward(self, x):
        """
        :param x: input embeddings 
        :return: output of decoder block, same shape as input
        """
        t = torch.cat([head(x) for head in self.heads], dim=-1)
        t = self.W_O(t)
        t = self.layernorm(t + x)
        t = self.FFN(t) 
        t = self.layernorm(t + x)

        return t#, [attn]


    def batch(self, b):
        self.b = b
        self.penc.batched = b


    def expand(self, d_mnew, d_inew):

        # TODO: / room for future experiments, how can we expand this ffn to not erase it everytime we expand?
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(self.num_heads*d_mnew, d_mnew),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(d_mnew, self.vocab_size)
        )
        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew-self.d_model, self.d_model)], dim=0)
        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew, d_mnew-self.d_model)], dim=1)
        for i in range(self.d_model+1, d_mnew):
            self.W_O.weight.data[i][i] = 1

        for head in self.heads:
            head.expand(d_mnew, d_inew)

        self.d_model = d_mnew
        self.d_internal = d_inew




def training_loop(model, data, dev=None, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    results = []
    avg_loss = []
    for t in range(num_epochs):
        loss_fnc = nn.NLLLoss()
        model.train()
        l = 0.
        for i, (d, label) in enumerate(data):
            py, x = model(d)
            loss = loss_fnc(py.view(-1,3), label.view(-1))

            model.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()

        # print("epoch {}:\t".format(t), decode(model, dev))
        avg_loss.append(l/len(data))

        if dev != None:
            model.eval()
            r = decode(model, dev)
            results.append(r[-1])

    
    model.train()

    if dev != None:
        return model, avg_loss, results 

    return model, avg_loss



def model_run(model_args, epochs, num_trials, transfer_ratio, do_logging):
    data = load_dataset('stanfordnlp/sst2')
    validation = data['validation']
    test = data['test']
    train = data['train']

    prev_args = None

    num_epochs = 50
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    train = train.map(lambda ex: tokenizer(ex['sentence'], padding='max_length', truncation=True), batched=True)
    validation = validation.map(lambda ex: tokenizer(ex['sentence'], padding='max_length', truncation=True), batched=True)
    test = test.map(lambda ex: tokenizer(ex['sentence'], padding='max_length', truncation=True), batched=True)



    for args in model_args:
        if prev_args == None:
            prev_args = args
            continue
        res_std = []
        res_tran = []
        res_full_train = []
        pprev_args = []
        transfer_train = []
        full_train = []
        transfer_full_train = []
        
        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []

        for t in tqdm.tqdm(range(num_trials)):
            # small model for BUS
            # cannot pass str to model, embeddings expect indices not str's
            model = Decoder(**prev_args).to(DEVICE)
            print(model(train['sentence'][0]))
            model, l1, r1 = training_loop(model, train, validation, num_epochs*transfer_ratio)
            loss1.append(l1)


            # larger model for BUS
            m = MultiHeadTransformer(**args).to(DEVICE)
            m.BUS(model)
            m1, l2, r2 = training_loop(m, train, validation, num_epochs)
            loss2.append(l1+l2[:num_epochs*(1-transfer_ratio)])
            res_tran.append(r1+r2[:num_epochs*(1-transfer_ratio)])
            res_full_train.append(r2)
            # TODO: do pytorch logging instead this is so silly :P






if __name__ == "__main__":
    model_args = [
        # {'vocab_size':27, 'num_positions':20, 'd_model':16, 'd_internal':8, 'num_classes':3, 'num_layers':1},    # these two model sizes are too small to transfer information
        # {'vocab_size':27, 'num_positions':20, 'd_model':32, 'd_internal':16, 'num_classes':3, 'num_layers':1},
        # {'vocab_size':27, 'num_positions':20, 'd_model':64, 'd_internal':32, 'num_classes':3, 'num_layers':1},
        {'vocab_size':2000, 'num_positions':200, 'd_model':128, 'd_internal':64, 'num_classes':2, 'num_heads':1},
        {'vocab_size':2000, 'num_positions':200, 'd_model':256, 'd_internal':128, 'num_classes':2, 'num_heads':1},
        # {'vocab_size':27, 'num_positions':20, 'd_model':512, 'd_internal':256, 'num_classes':3, 'num_layers':1},
    ]

    model_run(model_args, epochs=50, num_trials=10, do_logging=True, transfer_ratio=0.15)






