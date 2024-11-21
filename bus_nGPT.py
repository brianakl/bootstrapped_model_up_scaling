import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# RoPE from GPTneo
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=-2):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Normalization for nGPT model
@torch.jit.script
def justnorm(x):
    ret = x / x.norm(p=2, dim=-1, keepdim=True)
    return ret


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_internal, rope_percentage=1.0):
        super().__init__()

        self.tril = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))

        self.qkv = torch.nn.Linear(d_model, 3*d_internal, bias=False)
        self.d_model = d_model
        self.d_internal = d_internal

        self.SoftMax = torch.nn.Softmax(dim=-1)
        self.rope = Rotary(d_internal)
        self.rope_percentage = rope_percentage
        
        self.sqk_init_value = 1.0       
        self.sqk_init_scaling = d_model ** -0.5
        # self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(d_internal, dtype=torch.float32))
        self.sqk = self.sqk_init_scaling*torch.ones(self.d_internal, dtype=torch.float32, device=DEVICE)


    def forward(self, input_vecs):
        """
        args:
            input_vecs [batch size, seq_len, d_model]
                : input vectors
        
        """
        B, T, C = input_vecs.shape

        qkv = self.qkv(input_vecs)
        Q, K, V = torch.split(qkv, qkv.size(2) // 3, dim=-1)
        cos, sin = self.rope(Q)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        Q = justnorm(Q)*self.sqk
        K = justnorm(K)*self.sqk

        if self.training:
            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0, is_causal=True, scale=C**0.5)
        else:
            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0, is_causal=True, scale=C**0.5)
        return out


    def expand(self, d_mnew, d_inew):

        del self.rope
        self.rope = Rotary(d_inew)

        W_Q, W_K, W_V = torch.split(self.qkv.weight.data, self.d_internal, dim=0)

        W_Q = torch.cat([W_Q, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE).normal_(0,std)], dim=0)
        W_Q = torch.cat([W_Q, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE).normal_(0,std)], dim=1)
        # TODO: this is probably wrong/inaccurate
        #       I suspect that 1 might be too high and throw off the learning 
        #       especially with the nGPT model.
        #       maybe just remove it?
        for i in range(self.d_internal, d_inew):
            W_Q[i][i] = 1.

        W_K = torch.cat([W_K, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE).normal_(0,std)], dim=0)
        W_K = torch.cat([W_K, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE).normal_(0,std)], dim=1)
        for i in range(self.d_internal, d_inew):
            W_K[i][i] =  1.

        W_V = torch.cat([W_V, torch.zeros(d_inew - self.d_internal, self.d_model, device=DEVICE).normal_(0,std)], dim=0)
        W_V = torch.cat([W_V, torch.zeros(d_inew, d_mnew - self.d_model, device=DEVICE).normal_(0,std)], dim=1)
        for i in range(self.d_internal, d_inew):
            W_V[i][i] = 1.

        self.qkv.weight.data = torch.cat([W_Q, W_K, W_V], dim=0)

        self.sqk_init_scaling = d_mnew ** -0.5
        self.sqk.data = torch.cat([self.sqk.data, self.sqk_init_scaling*torch.ones(d_inew - self.d_internal, dtype=torch.float32, device=DEVICE)], dim=-1)

        self.d_internal = d_inew
        self.d_model = d_mnew

        self.normalize()

    def normalize(self):
        W_Q, W_K, W_V = torch.split(self.qkv.weight.data, self.d_internal, dim=0)
        W_Q = justnorm(W_Q)
        W_K = justnorm(W_K)
        W_V = justnorm(W_V)
        self.qkv.weight.data = torch.cat([W_Q, W_K, W_V], dim=0)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_model//num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_internal) for _ in range(num_heads)])
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 4*d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4*d_model, self.d_model),
        )
        self.Wu = torch.nn.Linear(d_model, 2 * 4 * d_model)
        self.Wv = torch.nn.Linear(4*d_model, d_model)
        self.silu = torch.nn.SiLU()
        self.W_O = torch.nn.Linear(d_model, d_model, False)

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1. 
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32))
        # self.mlp_alpha = self.mlp_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32, device=DEVICE)

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * d_model, dtype=torch.float32))
        # self.suv = self.suv_init_scaling*torch.ones(2 * 4 * d_model, dtype=torch.float32, device=DEVICE)
        self.sqdm = d_model ** 0.5

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = d_model ** -0.5 
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32))
        # self.attn_alpha = self.attn_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32, device=DEVICE)


    def forward(self, x):
        """
        :param x: input embeddings
        :return: output of decoder block, same shape as input
        """
        t = x
        t = torch.cat([head(t) for head in self.heads], dim=-1)

        t = justnorm(t)

        t = self.W_O(t)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = justnorm(x) 
        B_norm = justnorm(t)
                
        res = A_norm + lr * (B_norm - A_norm)
        x = justnorm(res)


        uv = self.Wu(x)
        suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.d_model ** 0.5))) 
        uv = suv * uv
        t = justnorm(uv)
        

        u, v = torch.chunk(t, 2, dim=-1)
        res = u * self.silu(v)

        t = self.Wv(res)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)
        A_norm = justnorm(x)
        B_norm = justnorm(t)

        res = A_norm + lr * (B_norm - A_norm)
        res = justnorm(res)

        return res


    def expand(self, d_mnew, d_inew):
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(d_mnew, 4*d_mnew),
            torch.nn.GELU(),
            torch.nn.Linear(4*d_mnew, d_mnew),
        )
        self.sqdm = d_mnew ** 0.5
        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew-self.d_model, self.d_model, device=DEVICE).normal_(0,std)], dim=0)
        self.W_O.weight.data = torch.cat([self.W_O.weight.data, torch.zeros(d_mnew, d_mnew-self.d_model,  device=DEVICE).normal_(0,std)], dim=1)
        # TODO: apply same update as in attention head class
        for i in range(self.d_model+1, d_mnew):
            self.W_O.weight.data[i][i] = 1

        for head in self.heads:
            head.expand(d_mnew, d_inew)

        # TODO: maybe expand?       | low priority since linear layers learn pretty fast
        self.Wu = torch.nn.Linear(d_mnew, 2 * 4 * d_mnew)
        self.Wv = torch.nn.Linear(4*d_mnew, d_mnew)

        self.mlp_alpha.data = torch.cat([self.mlp_alpha.data, self.mlp_alpha_init_scaling*torch.ones(d_mnew - self.d_model, dtype=torch.float32, device=DEVICE)], dim=-1)
        self.suv.data = torch.cat([self.suv.data, self.suv_init_scaling*torch.ones(2 * 4 * (d_mnew - self.d_model), dtype=torch.float32, device=DEVICE)],dim=-1)
        
        self.attn_alpha_init_scaling = d_mnew ** -0.5 
        self.attn_alpha.data = torch.cat([self.attn_alpha.data, self.attn_alpha_init_scaling*torch.ones(d_mnew - self.d_model, dtype=torch.float32, device=DEVICE)], dim=-1)

        self.d_model = d_mnew
        self.d_internal = d_inew
        self.normalize()

    def normalize(self):
        self.W_O.weight.data =justnorm(self.W_O.weight.data)
        self.Wu.weight.data = justnorm(self.Wu.weight.data)
        self.Wv.weight.data = justnorm(self.Wv.weight.data)
        self.mlp_alpha.data = justnorm(self.mlp_alpha.data)
        self.suv.data = justnorm(self.suv.data)
        self.attn_alpha.data = justnorm(self.attn_alpha.data)

        for h in self.heads:
            h.normalize()


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.blocks = torch.nn.ModuleList([TransformerLayer(d_model, num_heads) for _ in range(num_layers)])

        self.sm = torch.nn.LogSoftmax(dim=-1)
        self.lin = torch.nn.Linear(d_model, vocab_size)
        self.dout = torch.nn.Dropout(0.1)

        self.embeddings = torch.nn.Embedding(vocab_size, d_model, device=DEVICE)
        torch.backends.cuda.enable_flash_sdp(True)

        if torch.backends.cuda.flash_sdp_enabled(): print("Flash attention enabled")

        self.pretraining = True
        self.output_classes = 2

        self.sz_init = 1.
        self.sz_scale = d_model ** -0.5
        self.sz = torch.nn.Parameter(self.sz_init * torch.ones(vocab_size, dtype=torch.float32))


        self.tophat = torch.nn.Sequential(          # LM head for finetuning
            torch.nn.Linear(d_model, 4*d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4*d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, self.output_classes),
        )


    def forward(self, x):
        if self.pretraining:
            x = self.embeddings(x) 
            x = self.dout(x)
            t = x
            for head in self.blocks:
                t = head(t)

            sz = self.sz * (self.sz_init/self.sz_scale)
            t = sz * self.lin(t)

            return self.sm(t)
        else:
            with torch.no_grad():
                x = self.embeddings(x) 
                t = x
                for head in self.blocks:
                    t = head(t)

            return self.tophat(t)


    def expand(self, d_mnew):
        d_inew = d_mnew // self.num_heads
        self.lin = torch.nn.Linear(d_mnew, self.vocab_size)

        for block in self.blocks:
            block.expand(d_mnew, d_inew)

        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.cat([
                self.embeddings.weight, 
                torch.ones(self.vocab_size, d_mnew-self.d_model, device=DEVICE).normal_(mean=0, std=std)
                ], dim=1))

        self.d_model = d_mnew
        self.d_internal = d_inew
        self.to(DEVICE)


    def normalize(self):
        for block in self.blocks:
            block.normalize()

        self.embeddings.weight.data = justnorm(self.embeddings.weight.data)
        self.lin.weight.data = justnorm(self.lin.weight.data)


