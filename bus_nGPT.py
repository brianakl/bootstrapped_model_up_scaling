import torch
import torch.nn.functional as F

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

class AttentionHead(torch.nn.Module):
    def __init__(self, d_model, d_internal, num_heads, rope_percentage=1.0):
        super().__init__()

        stdev = 1/((d_model+d_internal)**0.5)
        self.qkv = torch.nn.Parameter(torch.ones((3*d_internal, d_model), device=DEVICE).uniform_(-stdev,stdev), requires_grad=True)
        self.qkv.data = justnorm(self.qkv.data)
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads

        self.rope = Rotary(d_internal)
        self.rope_percentage = rope_percentage
        
        self.sqk_init_value = 1.0       
        self.sqk_init_scaling = d_model ** -0.5
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(d_internal, dtype=torch.float32), requires_grad=True)


    # @torch.autocast(device_type=DEVICE)
    def forward(self, input_vecs):
        """
        args:
            input_vecs [batch size, seq_len, d_model]
                : input vectors
        
        """
        B, T, C = input_vecs.shape

        qkv = F.linear(input_vecs, self.qkv)
        Q, K, V = torch.split(qkv, qkv.size(2) // 3, dim=-1)
        cos, sin = self.rope(Q)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling))#.view(1, 1, self.num_heads, self.d_model// self.num_heads)
        Q = justnorm(Q)*sqk
        K = justnorm(K)*sqk

        if self.training:
            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0, is_causal=True, scale=C**0.5)
        else:
            out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0, is_causal=True, scale=C**0.5)

        return out


    def expand(self, d_mnew, d_inew):

        del self.rope
        self.rope = Rotary(d_inew)
        exp_ratio = self.d_model/d_mnew

        self.qkv.data *= exp_ratio
        W_Q, W_K, W_V = torch.chunk(self.qkv.data, 3, dim=0)

        W_Q = W_Q.repeat(2,2)[:d_inew, :d_mnew].clone()
        W_K = W_K.repeat(2,2)[:d_inew, :d_mnew].clone()
        W_V = W_V.repeat(2,2)[:d_inew, :d_mnew].clone()

        self.qkv.data = torch.cat([W_Q, W_K, W_V], dim=0)
        self.sqk_init_scaling = d_mnew ** -0.5
        self.sqk.data = self.sqk.data.repeat(1,2)[0][:d_inew].clone() * exp_ratio

        self.d_internal = d_inew
        self.d_model = d_mnew


    def normalize(self):
        W_Q, W_K, W_V = torch.chunk(self.qkv.data, 3, dim=0)
        W_Q = justnorm(W_Q)
        W_K = justnorm(W_K)
        W_V = justnorm(W_V)
        self.qkv.data = torch.cat([W_Q, W_K, W_V], dim=0)


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_model//num_heads
        self.num_heads = num_heads

        self.heads = torch.nn.ModuleList([AttentionHead(d_model, self.d_internal, num_heads) for _ in range(num_heads)])
        stdev = 1/((4*d_model + d_model)**0.5)
        self.Wu=torch.nn.Parameter(torch.zeros((2*4*d_model, d_model), device=DEVICE).uniform_(-stdev,stdev), requires_grad=True)
        self.Wu.data = justnorm(self.Wu.data)
        self.Wv = torch.nn.Parameter(torch.zeros((d_model, 4*d_model), device=DEVICE).uniform_(-stdev,stdev), requires_grad=True)
        self.Wv.data = justnorm(self.Wv.data)
        self.silu = torch.nn.SiLU()
        self.W_O =  torch.nn.Parameter(torch.zeros((d_model, d_model), device=DEVICE).uniform_(-stdev,stdev), requires_grad=True)
        self.W_O.data = justnorm(self.W_O.data)

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1. 
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32), requires_grad=True)

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * d_model, dtype=torch.float32), requires_grad=True)
        self.sqdm = d_model ** 0.5

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = d_model ** -0.5 
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(d_model, dtype=torch.float32), requires_grad=True)

    # @torch.autocast(device_type=DEVICE)
    def forward(self, x):
        """
        :param x: input embeddings
                [batch size, context length, d_model]
        :return: output of decoder block, same shape as input
        """
        t = x
        t = torch.cat([head(t) for head in self.heads], dim=-1)

        t = F.linear(t, self.W_O)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = justnorm(x) 
        B_norm = justnorm(t)

        res = A_norm + lr * (B_norm - A_norm)
        x = justnorm(res)


        uv = F.linear(x, self.Wu)
        suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.d_model ** 0.5))) 
        uv = suv * uv
        t = justnorm(uv)
        

        u, v = torch.chunk(t, 2, dim=-1)
        res = u * F.silu(v) #self.silu(v)

        t = F.linear(res, self.Wv)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)
        A_norm = justnorm(x) 
        B_norm = justnorm(t)

        res = A_norm + lr * (B_norm - A_norm)
        res = justnorm(res)

        return res

    def expand(self, d_mnew, d_inew):
        exp_ratio = self.d_model/d_mnew
        self.W_O.data *= exp_ratio
        self.sqdm = d_mnew ** 0.5
        self.W_O.data = self.W_O.data.repeat(2,2)
        self.W_O.data = self.W_O.data[:d_mnew, :d_mnew].clone()

        for head in self.heads:
            head.expand(d_mnew, d_inew)

        self.Wv.data *= exp_ratio
        self.Wv.data = self.Wv.data.repeat(2,2)
        self.Wv.data = self.Wv.data[:d_mnew, :4*d_mnew].clone()

        self.Wu.data *= exp_ratio
        b1, b2 = self.Wu.data.chunk(2)
        b1 = b1.repeat(2,2)[:4*d_mnew, :d_mnew]
        b2 = b2.repeat(2,2)[:4*d_mnew, :d_mnew]

        b = torch.cat([b1,b2])
        self.Wu.data = b.clone()

        self.mlp_alpha.data *= exp_ratio
        self.mlp_alpha.data = self.mlp_alpha.data.repeat(1,2)[0][:d_mnew].clone()
        self.suv.data *= exp_ratio
        s1, s2 = self.suv.data.clone().chunk(2)
        s1 = s1.repeat(1,2)[:, :4*d_mnew]
        s2 = s2.repeat(1,2)[:, :4*d_mnew]
        self.suv.data = torch.cat([s1,s2], dim=-1)[0].clone()
        
        self.attn_alpha_init_scaling = d_mnew ** -0.5 
        self.attn_alpha.data *= exp_ratio
        self.attn_alpha.data = self.attn_alpha.data.repeat(1,2)[0][:d_mnew].clone()

        self.d_model = d_mnew
        self.d_internal = d_inew

    def normalize(self, b = True):

        self.W_O.data =justnorm(self.W_O)
        self.Wu.data = justnorm(self.Wu)
        self.Wv.data = justnorm(self.Wv)

        if b:
            for h in self.heads:
                h.normalize()


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.blocks = torch.nn.ModuleList([TransformerLayer(d_model, num_heads) for _ in range(num_layers)])

        stdev = 1/((d_model)**0.5)
        self.lin = torch.nn.Parameter(torch.zeros((vocab_size, d_model), device=DEVICE).uniform_(-stdev,stdev))

        self.embeddings = torch.nn.Embedding(vocab_size, d_model, device=DEVICE)
        torch.backends.cuda.enable_flash_sdp(True)

        if torch.backends.cuda.flash_sdp_enabled(): print("Flash attention enabled")

        self.pretraining = True
        self.output_classes = 2

        self.sz_init = 1.
        self.sz_scale = d_model ** -0.5
        self.sz = torch.nn.Parameter(self.sz_scale * torch.ones(vocab_size, dtype=torch.float32))

    @torch.autocast(device_type=DEVICE)
    def forward(self, input:torch.Tensor):
        if self.pretraining:
            x = self.embeddings(input) 

            for head in self.blocks:
                x = head(x) + x

            b = F.linear(x, self.lin)
            sz = self.sz * (self.sz_init/self.sz_scale)
            t = b * sz

            return t#F.softmax(t, dim=-1)
        else:
            # Headless mode
            with torch.no_grad():
                x = self.embeddings(input) 

                for head in self.blocks:
                    x = head(x) + x

                return x



    def expand(self, d_mnew):
        d_inew = d_mnew // self.num_heads
        exp_ratio = self.d_model/d_mnew
        self.lin.data *= exp_ratio
        self.lin.data = self.lin.data.repeat(1,2)[:, :d_mnew]
        self.embeddings.weight.data = self.embeddings.weight.data.repeat(1,2)[:, :d_mnew].clone()

        self.sz_scale = d_mnew ** -0.5
        for block in self.blocks:
            block.expand(d_mnew, d_inew)

        self.d_model = d_mnew
        self.d_internal = d_inew
        self.normalize()
        return self


    def normalize(self, b = True):

        self.embeddings.weight.data = justnorm(self.embeddings.weight.data)
        self.lin.data = justnorm(self.lin.data)
        if b:
            for block in self.blocks:
                block.normalize()

