import torch
import torch.nn as nn


def rotate_half(x):
    # (...,head_dim) -> the same vector with its two halves swapped and the first negated
    # this is what turns a multiply into a 2d rotation
    x1, x2 = x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)


def apply_rotary(x,cos,sin):
    # x: (N,heads,seq_len,head_dim), cos/sin: (seq_len,head_dim) -> broadcast over N and heads
    return x*cos + rotate_half(x)*sin


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Instead of adding a position vector to the input, every 2d slice of a
    query/key vector is rotated by an angle proportional to its position.
    The dot product between a query at position i and a key at position j
    then depends only on i-j, so attention sees relative distance.
    """

    def __init__(self,head_dim,max_len=2048,theta=10000.0):
        super(RotaryEmbedding,self).__init__()

        assert head_dim % 2 == 0, "RoPE rotates pairs of dims, so head_dim must be even"

        # one frequency per pair of dims, geometrically spaced:
        # fast rotation in the first dims, slow in the last ones
        inv_freq = 1.0/(theta**(torch.arange(0,head_dim,2).float()/head_dim))

        # angle of every (position,pair): (max_len,head_dim/2)
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos,inv_freq)

        # duplicated so it lines up with rotate_half's two halves: (max_len,head_dim)
        emb = torch.cat((freqs,freqs),dim=-1)

        # buffers, not parameters: RoPE has nothing to learn
        self.register_buffer("cos_cached",emb.cos(),persistent=False)
        self.register_buffer("sin_cached",emb.sin(),persistent=False)
        self.max_len = max_len

    def forward(self,seq_len,offset=0):
        # offset lets a cached-generation step ask for positions further along
        assert offset + seq_len <= self.max_len, f"sequence longer than max_len={self.max_len}"
        return (self.cos_cached[offset:offset+seq_len],
                self.sin_cached[offset:offset+seq_len])


class Attention(nn.Module):
    def __init__(self,embed_size,heads,max_len=2048):
        super(Attention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert (
            self.head_dim * heads == embed_size), "Embed size should be divisible by hears"

        self.query = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.key = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)

        self.rotary = RotaryEmbedding(self.head_dim,max_len)

        self.fc_out = nn.Linear(heads*self.head_dim,embed_size)

    def forward(self, query, keys, values,mask=None,offset=0):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split the embedding into self.heads pieces
        values = values.reshape(N,value_len,self.heads,self.head_dim)
        keys = keys.reshape(N,key_len,self.heads,self.head_dim)
        queries = query.reshape(N,query_len,self.heads,self.head_dim)

        # move heads next to the batch dim so it acts as a batch dim for the matmuls
        # (N,len,heads,head_dim) -> (N,heads,len,head_dim)
        values = self.values(values).transpose(1,2)
        keys = self.key(keys).transpose(1,2)
        queries = self.query(queries).transpose(1,2)

        # rotate queries and keys by their position. values are NOT rotated:
        # position belongs in the scores, not in the content being summed
        cos, sin = self.rotary(query_len,offset)
        queries = apply_rotary(queries,cos,sin)
        cos, sin = self.rotary(key_len,offset)
        keys = apply_rotary(keys,cos,sin)

        # (N,heads,query_len,head_dim) @ (N,heads,head_dim,key_len)
        # energy: (N,heads,query_len,key_len)
        energy = queries @ keys.transpose(-2,-1)

        if mask is not None:
            energy = energy.masked_fill(mask == 0,float("-1e20"))

        attention = torch.softmax(energy/(self.head_dim**0.5),dim=3)

        # (N,heads,query_len,key_len) @ (N,heads,value_len,head_dim)
        # out: (N,heads,query_len,head_dim) -> (N,query_len,heads,head_dim) -> flatten heads
        out = (attention @ values).transpose(1,2).reshape(
            N,query_len,self.heads*self.head_dim)

        return self.fc_out(out)
