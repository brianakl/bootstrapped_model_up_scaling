import tqdm
import numpy
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()

        self.W_Q = torch.nn.Linear(d_model, d_internal, False)
        self.W_K = torch.nn.Linear(d_model, d_internal, False)
        self.W_V = torch.nn.Linear(d_model, d_model, False)

        self.SoftMax = torch.nn.Softmax(dim=-1)


        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_internal),
            torch.nn.ReLU(), 
            torch.nn.Dropout(),
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

class Transformer(nn.Module):
    def __init__(self, d_model, d_internal, num_classes, **args):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        self.tformer = TransformerLayer(d_model, d_internal)
        self.Softmax = torch.nn.LogSoftmax(dim=-1)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(d_model, num_classes)
        )
        self.b = False
        # self.penc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=self.b)
        # self.embed = torch.nn.Embedding(vocab_size, d_model).to(DEVICE)

        self.double()


    def forward(self, x):
        """
        :param indices: list of input indices
        :return: A tuple of softmax log probabilities and a list of the attention maps 
        """
        # t = self.embed(indices.to(DEVICE))
        # t = self.penc(t)
        # t = t.to(torch.float64)
        t, attn = self.tformer(x)
        x = self.FFN(t)
        x = self.Softmax(x)
        return x, [attn]


    def batch(self, b):
        self.b = b
        self.penc.batched = b



    def extrap(self, model, method='onehot'):

        if method == "onehot":

            # for one hot we simply add the corresponding one hot vector to the dimensions of the vector

            Q = model.tformer.W_Q.weight.data
            z = torch.zeros((self.d_internal - model.d_internal, model.tformer.d_model)).to(DEVICE)
            # z = torch.randn((self.d_internal - model.d_internal, model.tformer.d_model)).to(DEVICE)
            Q = torch.cat((Q,z), dim=0)
            z = torch.zeros((self.d_model - model.d_model, self.d_model - model.d_model)).to(DEVICE)
            # z = torch.randn((self.d_model - model.d_model, self.d_model - model.d_model)).to(DEVICE)
            Q = torch.cat((Q,z), dim=-1)
            for i in range(self.d_internal):
                Q[i][i] = 1 if Q[i][i] == 0 else Q[i][i]
            

            V = model.tformer.W_V.weight.data
            z = torch.zeros((self.d_model - model.d_model, model.tformer.d_model)).to(DEVICE)
            # z = torch.randn((self.d_model - model.d_model, model.tformer.d_model)).to(DEVICE)
            V = torch.cat((V,z), dim=0)
            z = torch.zeros((self.d_model, model.tformer.d_model)).to(DEVICE)
            # z = torch.randn((self.d_model, model.tformer.d_model)).to(DEVICE)
            V = torch.cat((V,z), dim=-1)
            for i in range(self.d_model):
                V[i][i] = 1 if V[i][i] == 0 else V[i][i]

            K = model.tformer.W_K.weight.data
            z = torch.zeros((self.d_internal - model.d_internal, model.tformer.d_model)).to(DEVICE)
            # z = torch.randn((self.d_internal - model.d_internal, model.tformer.d_model)).to(DEVICE)
            K = torch.cat((K,z), dim=0)
            z = torch.zeros((self.tformer.d_internal, model.tformer.d_model)).to(DEVICE)
            # z = torch.randn((self.tformer.d_internal, model.tformer.d_model)).to(DEVICE)
            K = torch.cat((K,z), dim=-1)
            for i in range(self.d_internal):
                K[i][i] = 1 if K[i][i] == 0 else K[i][i]


            self.tformer.W_Q.weight.data = Q
            self.tformer.W_K.weight.data = K
            self.tformer.W_V.weight.data = V





class MultiHeadTransformer(nn.Module):
    def __init__(self, num_heads, vocab_size, num_positions, d_model, d_internal, num_classes) -> None:
        """
        :param num_heads: number of heads 
        :param vocab_size: size of vocab for embeddings
        :param num_positions: context length
        :param d_model: d_model of transformer
        :param d_internal: d_internal of transformer
        :param num_classes: number of classes for task (sentiment analysis = 2)
        """
        super().__init__()
        self.num_heads = num_heads
        self.penc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=True)
        self.embed = torch.nn.Embedding(vocab_size, d_model).to(DEVICE)
        self.heads = [Transformer(d_model, d_internal, num_classes) for _ in range(num_heads)]
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(d_model*num_heads, d_model),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.Softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embed(x) + self.penc(x)
        x = torch.cat([self.heads[i](x) for i in range(self.num_heads)], dim=-1)
        x = self.nn(x)
        return self.Softmax(x)

    def BUS(self, model):
        """
        :param model: smaller model that will be used as a basis to perform BUS (assuming same number of heads)
        """
        for i in range(self.num_heads):
            self.heads[i].extrap(model.heads[i])



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
            model = MultiHeadTransformer(**prev_args).to(DEVICE)
            print(model(train['sentence'][0]))





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






