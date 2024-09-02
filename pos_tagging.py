from transfer import Transformer, TransformerLayer, training_loop
import tqdm
import numpy
import pytorch
from datasets import load_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pos_tagging(model_args, epochs, num_trials, transfer_ratio, do_logging):
    data = load_dataset('stanfordnlp/sst2')
    validation = data['validation']
    test = data['test']
    train = data['train']

    prev_args = None


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
            model = Transformer(**prev_args).to(DEVICE)

if __name__ == "__main__":
    model_args = [
        # {'vocab_size':27, 'num_positions':20, 'd_model':16, 'd_internal':8, 'num_classes':3, 'num_layers':1},    # these two model sizes are too small to transfer information
        # {'vocab_size':27, 'num_positions':20, 'd_model':32, 'd_internal':16, 'num_classes':3, 'num_layers':1},
        # {'vocab_size':27, 'num_positions':20, 'd_model':64, 'd_internal':32, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':128, 'd_internal':64, 'num_classes':3, 'num_layers':1},
        {'vocab_size':27, 'num_positions':20, 'd_model':256, 'd_internal':128, 'num_classes':3, 'num_layers':1},
        # {'vocab_size':27, 'num_positions':20, 'd_model':512, 'd_internal':256, 'num_classes':3, 'num_layers':1},
    ]







