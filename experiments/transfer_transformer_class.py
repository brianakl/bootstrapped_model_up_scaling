import torch 
import torch.nn as nn
import numpy as np
from torch import full, optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, num_positions: int=20, batched=True):
        super().__init__()

        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched
        self.d_model = d_model
        self.num_positions = num_positions
        
        self.sinu = torch.zeros((num_positions, d_model)).to(DEVICE)
            


        for pos in range(num_positions):
            for m in range(d_model):
                if m%2 == 0:
                    self.sinu[pos][m] += torch.sin(torch.tensor(pos/(10000**((2*m)/self.d_model))))
                else:
                    self.sinu[pos][m] += torch.cos(torch.tensor(pos/(10000**((2*m)/self.d_model))))


    def forward(self, x):
        # input_size = x.shape[-2]
        # indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)

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
        self.embed = torch.nn.Embedding(vocab_size, d_model).to(DEVICE)

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
            

            Q = torch.matmul(z, torch.transpose(model.tformer.W_Q.weight.data, -1, -2))
            Q = torch.matmul(Q, torch.transpose(z1, -1, -2))
            self.tformer.W_Q.weight.data = torch.transpose(Q, -1, -2)

            K = torch.matmul(z, torch.transpose(model.tformer.W_K.weight.data, -1, -2))
            K = torch.matmul(z1, torch.transpose(K, -1, -2))
            self.tformer.W_K.weight.data = K

            V = torch.matmul(z, torch.transpose(model.tformer.W_V.weight.data, -1, -2))     # especially this one for V
            V = torch.matmul(z, torch.transpose(V, -1, -2))
            self.tformer.W_V.weight.data = torch.transpose(V, -1, -2)



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






    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of softmax log probabilities and a list of the attention maps 
        """
        t = self.embed(indices.to(DEVICE))
        t = self.penc(t)
        t = t.to(torch.float64)
        t, attn = self.tformer(t)
        x = self.FFN(t)
        x = self.Softmax(x)
        return x, [attn]


    def batch(self, b):
        self.b = b
        self.penc.batched = b




def get_letter_count_output(input: str, count_only_previous: bool=True) -> np.array:
    """
    :param input: The string
    :param count_only_previous: True if we should only count previous occurences, False for all occurences
    :return: the output for the letter-counting task as a numpy array of 0s, 1s, and 2s
    """
    output = np.zeros(len(input))
    for i in range(0, len(input)):
        if count_only_previous:
            output[i] = min(2, len([c for c in input[0:i] if c == input[i]]))
        else:
            output[i] = min(2, len([c for c in input if c == input[i]])-1)
    return output



def read_example(file):
    """
    :param file: input file
    :return: A list of the lines in the file, each exactly 20 characters long
    """
    all_lines = []
    for line in open(file):
        all_lines.append(line[:-1])     # remove the \n

    return all_lines



class SentenceData(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        self.n = len(data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])
 


def train_classifier(args, train:LetterCountingExample, dev:LetterCountingExample, extrap=None, num_epochs=10):

    model = Transformer(**args)
    if extrap != None:
        model.extrap(extrap, method='onehot')

    model.to(DEVICE)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    x_train = []
    y_train = []

    for i in range(len(train)):
        x_train.append(train[i].input_tensor.to(DEVICE))
        y_train.append(train[i].output_tensor.to(DEVICE))

    ds = SentenceData(x_train, y_train)

    data = DataLoader(ds, batch_size=16, shuffle=True)

    training_loop(model, data, num_epochs)


def training_loop(model, data, dev, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    results = []
    avg_loss = []
    for t in range(num_epochs):
        loss_fnc = nn.NLLLoss()
        # model.train()
        l = 0.
        for i, (d, label) in enumerate(data):
            py, x = model(d)
            loss = loss_fnc(py.view(-1,3), label.view(-1))

            model.zero_grad()
            loss.backward()
            optimizer.step()
            # l += loss.item()

        # print("epoch {}:\t".format(t), decode(model, dev))

        # avg_loss.append(l/len(data))
        # model.eval()
        # results.append(decode(model, dev)[-1])

    
    model.train()
    return model, results, avg_loss




def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples and prints the final accuracy
    :param model: the model that outputs the log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return: None    
    """
    num_correct = 0 
    num_total = 0

    if len(dev_examples) > 100:
        if do_print:
            print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.cpu().detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    if do_print:
        print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    return (num_correct, num_total, float(num_correct) / num_total)





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


































