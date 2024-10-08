import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionHead(torch.nn.Module):
    def __init__(self, d_model, d_internal, seq_len):
        super().__init__()

        self.W_Q = torch.nn.Linear(d_model, d_internal, False)
        self.W_K = torch.nn.Linear(d_model, d_internal, False)
        self.W_V = torch.nn.Linear(d_model, d_internal, False)

        self.Softmax = torch.nn.Softmax(dim=-1)


        self.d_model = d_model
        self.d_internal = d_internal
        self.tril = torch.tril(torch.ones())
        self.seq_len = seq_len


    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_K(x)

        attn = Q @ K.transpose(-2, -1) * C**-0.5
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = self.Softmax(attn) @  V
        
        return attn

    def expand(self, d_mnew, d_inew):
        self.W_Q.weight.data = torch.cat([self.W_Q.weight.data, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE)], dim=0)
        self.W_Q.weight.data = torch.cat([self.W_Q.weight.data, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE)], dim=1)
        for i in range(self.d_internal, d_inew):
            self.W_Q.weight.data[i][i] = self.W_Q.weight.data[i][i] if self.W_Q.weight.data[i][i] != 0 else 1

        self.W_K.weight.data = torch.cat([self.W_K.weight.data, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE)], dim=0)
        self.W_K.weight.data = torch.cat([self.W_K.weight.data, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE)], dim=1)
        for i in range(self.d_internal, d_inew):
            self.W_K.weight.data[i][i] = self.W_K.weight.data[i][i] if self.W_K.weight.data[i][i] != 0 else 1

        self.W_V.weight.data = torch.cat([self.W_V.weight.data, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE)], dim=0)
        self.W_V.weight.data = torch.cat([self.W_V.weight.data, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE)], dim=1)
        for i in range(self.d_internal, d_inew):
            self.W_V.weight.data[i][i] = self.W_V.weight.data[i][i] if self.W_V.weight.data[i][i] != 0 else 1

        self.d_internal = d_inew
        self.d_model = d_mnew 
        self.SoftMax = torch.nn.Softmax(dim=-1)


class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, seq_len):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_model // num_heads
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.seq_len = seq_len

        self.heads = torch.nn.ModuleList([AttentionHead(d_model, self.d_internal, seq_len) for _ in range(num_heads)])
        self.Softmax = torch.nn.Softmax(dim=-1)
        self.FFN = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_hidden),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_hidden, self.d_model),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
        )

        self.W_O = torch.nn.Linear(d_model, d_model, False)
        self.layernorm = torch.nn.LayerNorm(d_model)


    def forward(self, x):
        t = torch.cat([head(x) for head in self.heads], dim=-1)
        t = self.W_O(t)
        t = self.layernorm(t + x)
        t = self.FFN(t)
        t = self.layernorm(t + x)

        return t


    def expand(self, d_mnew):

        d_inew = d_mnew // self.num_heads


        self.FFN = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_hidden),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_hidden, self.d_model),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
        )

        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew-self.d_model, self.d_model, device=DEVICE)], dim=0)
        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew, d_mnew-self.d_model,  device=DEVICE)], dim=1)
        for i in range(self.d_model+1, d_mnew):
            self.W_O.weight.data[i][i] = 1

        self.layernorm = torch.nn.LayerNorm(d_mnew)

        for head in self.heads:
            head.expand(d_mnew, d_inew)

        self.d_model = d_mnew
        self.d_internal = d_inew



class Decoder(torch.nn.Module):
    def __init__(self, d_model, num_layers, d_hidden, vocab_size, num_heads, seq_len):
        super().__init__()
        self.num_layers = num_layers 
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.Softmax = torch.nn.LogSoftmax(dim=-1)
        self.dout = torch.nn.Dropout(0.1)
        self.blocks = torch.nn.ModuleList([Transformer(d_model, num_heads, d_hidden, seq_len) for _ in range(num_layers)])
        self.FFN = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_hidden),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(d_hidden, vocab_size),
                torch.nn.LogSoftmax(dim=-1),
        )

        self.embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = torch.nn.Embedding(seq_len, d_model)
        self.generate_pos_embed(d_model)

        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embeddings(x) + self.pos_embeddings(torch.arange(x.shape[-1]))
        x = self.dout(x)
        t = x
        for head in self.blocks:
            t = head(t) 

        t = self.FFN(t) 

        return t


    def generate_pos_embed(self, d_model):
        # TODO: make more efficient 
        pos_em = torch.zeros((self.seq_len, d_model))
        for pos in range(self.seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_em[pos][i] += torch.sin(torch.tensor(pos/(10000**(2*i/d_model))))
                else:
                    pos_em[pos][i] += torch.cos(torch.tensor(pos/(10000** (2*i/d_model))))

        self.pos_embedding = torch.nn.Embedding.from_pretrained(pos_em, freeze=True)
        

    def expand(self, d_mnew):
        d_inew = d_mnew // self.num_heads
        self.layernorm = torch.nn.LayerNorm(d_mnew)


        self.FFN = torch.nn.Sequential(
                torch.nn.Linear(d_mnew, self.d_hidden),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_hidden, self.vocab_size),
                torch.nn.LogSoftmax(dim=-1),
        )

        self.embeddings = torch.nn.Embedding.from_pretrained(torch.cat([self.embeddings.weight, torch.zeros(self.vocab_size, d_mnew-self.d_model, device=DEVICE).uniform_()], dim=1))
        self.generate_pos_embed(d_mnew)

        self.d_model = d_mnew

        for block in self.blocks:
            block.expand(d_mnew, d_inew)

class Norm(torch.nn.Module):
    def __init__(self, eps = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)



# RoFormer Embeddings
# pytorch implementation
class RoFormerEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = torch.nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps) if config.norm_type=="layer_norm" else Norm(eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=inputs_embeds.device
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings












