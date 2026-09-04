import torch
import torch.nn as nn
from src.tokenizer.tokenizer_default import Tokenizer
from src.attention.attention import Attention
from config.config import config


class DecoderBlock(nn.Module):
    def __init__(self,emb_size,heads,max_len,dropout=0.2):
        super(DecoderBlock,self).__init__()
        self.attention = Attention(emb_size,heads,max_len)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
                    nn.Linear(emb_size, 4 * emb_size),
                    nn.ReLU(),
                    nn.Linear(4 * emb_size, emb_size),
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        # sub-layer 1: attention, its own residual, then norm
        attention = self.attention(x,x,x,mask)
        x = self.norm1(x + self.dropout(attention))

        # sub-layer 2: feed-forward, its own residual, then norm
        # attention only mixes information between tokens; this is where
        # each token's features get transformed on their own
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out


class Decoder(nn.Module):
    def __init__(self,
                 emb_size = 128,
                 tokenizer = config.tokenizer,
                 heads = 8,
                 device = config.device,
                 num_layers = 1,
                 max_len = 512,
                 dropout = 0.2
                 ):
        super(Decoder,self).__init__()
        self.device = device
        self.tokenizer =  Tokenizer(tokenizer)
        self.vocab_size = len(self.tokenizer.tokenizer)
        self.emb_size = emb_size
        # one learnable vector per vocabulary entry: what the token means
        self.word_embedding = nn.Embedding(self.vocab_size,emb_size)
        # no position table here: RoPE adds the position inside Attention instead
        self.layers = nn.ModuleList(
            [DecoderBlock(emb_size,heads,max_len,dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # projects each position's hidden state back to vocabulary logits.
        # tied to the embedding table: the same matrix that turns a token id
        # into a vector is reused (transposed) to turn a vector back into
        # scores over the vocabulary, halving the parameters this layer needs
        self.fc_out = nn.Linear(emb_size,self.vocab_size)
        self.fc_out.weight = self.word_embedding.weight

        self.max_len = max_len
        self.to(self.device)

    def encode(self,text):
        # convenience for running raw text through the model outside of training,
        # where a DataLoader already hands forward() pre-tokenized batches
        tokens = torch.tensor(self.tokenizer.Encode(text),device=self.device)
        return tokens.unsqueeze(0)

    def forward(self,tokens):
        # tokens: (N,seq_len) of token ids, already batched
        out = self.dropout(self.word_embedding(tokens))
        # out carries no position information; RoPE puts it into the queries and keys

        seq_len = out.shape[1]
        # causal mask: token i may only attend to tokens 0..i, never the future
        mask = torch.tril(torch.ones(seq_len,seq_len,device=self.device))

        for layer in self.layers:
            out = layer(out,mask)

        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def generate_tokens(self,tokens,max_new_tokens=100,temperature=1.0,top_k=None,stop_ids=None):
        """Autoregressive sampling: feed the tokens, repeatedly predict one
        more from everything so far, append it, repeat.

        There is no KV cache, so each step reruns the whole growing sequence
        through every layer - simple and correct, but O(n^2) in tokens. Fine
        for a small model; a real serving setup caches past keys/values instead.

        tokens: (1,seq_len) -> returns (1,seq_len+n)
        """
        was_training = self.training
        self.eval()

        if stop_ids is None:
            stop_ids = {self.tokenizer.eos_id}

        for _ in range(max_new_tokens):
            # RoPE's table only covers max_len positions, so once the
            # sequence outgrows it we must feed just the most recent window
            context = tokens[:,-self.max_len:]

            logits = self(context)                              # (1,ctx_len,vocab_size)
            logits = logits[:,-1,:]/temperature                 # last position only: (1,vocab_size)

            if top_k is not None:
                values,_ = torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits < values[:,-1:]] = float("-inf")

            probs = torch.softmax(logits,dim=-1)
            next_token = torch.multinomial(probs,num_samples=1)  # (1,1)
            tokens = torch.cat([tokens,next_token],dim=1)

            if next_token.item() in stop_ids:
                break

        self.train(was_training)
        return tokens

    def generate(self,prompt,max_new_tokens=100,temperature=1.0,top_k=None,stop_ids=None):
        tokens = self.generate_tokens(
            self.encode(prompt),max_new_tokens,temperature,top_k,stop_ids
        )
        return self.tokenizer.Decode(tokens[0].tolist())



if __name__ == "__main__":
    dec = Decoder()
    print(dec(dec.encode("hello, how are you?")).shape)
