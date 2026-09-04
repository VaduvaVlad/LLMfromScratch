import os
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.transformer.transformer import Decoder
from src.tokenizer.tokenizer_default import USER_TOKEN, ASSISTANT_TOKEN

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "decoder.pt")
ULTRACHAT_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ultrachat_tokens.pt")


def download_dataset(path=DATA_PATH,url=DATA_URL):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        urllib.request.urlretrieve(url,path)
    with open(path,"r",encoding="utf-8") as f:
        return f.read()


def format_conversation(messages):
    """Flatten one UltraChat conversation into a single training string.

    <|user|>...<|assistant|>...<|user|>...<|assistant|>...<|endoftext|>

    The markers are what teach turn-taking: the model learns that text after
    <|assistant|> is its own to produce, and that a turn ends at the next marker.
    """
    parts = []
    for message in messages:
        marker = USER_TOKEN if message["role"] == "user" else ASSISTANT_TOKEN
        parts.append(marker + message["content"])
    return "".join(parts)


def load_ultrachat(tokenizer,num_conversations=20000,cache_path=ULTRACHAT_CACHE):
    """Tokenize N UltraChat conversations into one flat token stream.

    Cached to disk because tokenizing is slow and rerunning training
    shouldn't pay that cost twice.
    """
    if os.path.exists(cache_path):
        print(f"loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    from datasets import load_dataset

    print(f"downloading + tokenizing {num_conversations} conversations...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft")
    dataset = dataset.select(range(min(num_conversations,len(dataset))))

    texts = [format_conversation(ex["messages"]) + tokenizer.tokenizer.eos_token
             for ex in dataset]

    # batch encode: far faster than calling Encode() once per conversation
    encoded = tokenizer.tokenizer(texts)["input_ids"]

    ids = []
    for conversation in encoded:
        ids.extend(conversation)

    tokens = torch.tensor(ids,dtype=torch.long)

    os.makedirs(os.path.dirname(cache_path),exist_ok=True)
    torch.save(tokens,cache_path)
    print(f"cached {len(tokens):,} tokens to {cache_path}")
    return tokens


class TextDataset(Dataset):
    """Slides a fixed-size window over one long token stream.

    Each item is (x,y) where y is x shifted right by one token - the
    self-supervision signal for next-token prediction. Windows overlap and
    start at every valid index, so shuffling the DataLoader mixes far more
    training signal out of the same text than chopping it into disjoint blocks.
    """

    def __init__(self,tokens,seq_len,stride=1):
        # stride=1 gives a window starting at every token - maximum reuse of a
        # small corpus. stride=seq_len gives disjoint chunks, which is what you
        # want once the corpus is large: 25M tokens at stride 1 would be 25M
        # samples (a million+ steps per epoch) for no real benefit
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0,(len(self.tokens) - self.seq_len - 1)//self.stride + 1)

    def __getitem__(self,i):
        start = i*self.stride
        x = self.tokens[start:start+self.seq_len]
        y = self.tokens[start+1:start+self.seq_len+1]
        return x,y


def train(model,dataloader,optimizer,loss_fn,epochs,device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step,(x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)

            logits = model(x)                                        # (N,seq_len,vocab_size)
            loss = loss_fn(logits.reshape(-1,logits.size(-1)),y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dataloader)}: loss {loss.item():.4f}")

        print(f"epoch {epoch} done: avg loss {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    seq_len = 512
    batch_size = 36
    epochs = 10
    lr = 3e-4
    num_conversations = 20000

    # bigger than the shakespeare run: holding a conversation needs far more
    # capacity than mimicking verse structure
    model_config = dict(emb_size=384,heads=6,num_layers=6,max_len=seq_len)
    model = Decoder(**model_config)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    tokens = load_ultrachat(model.tokenizer,num_conversations)
    dataset = TextDataset(tokens,seq_len,stride=seq_len)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train(model,dataloader,optimizer,loss_fn,epochs,model.device)

    # save weights + the constructor args needed to rebuild the same
    # architecture, so inference doesn't have to guess or hardcode them
    os.makedirs(os.path.dirname(CHECKPOINT_PATH),exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model_config},CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")
