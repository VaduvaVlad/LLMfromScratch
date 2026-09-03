import os
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.transformer.transformer import Decoder

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "decoder.pt")


def download_dataset(path=DATA_PATH,url=DATA_URL):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        urllib.request.urlretrieve(url,path)
    with open(path,"r",encoding="utf-8") as f:
        return f.read()


class TextDataset(Dataset):
    """Slides a fixed-size window over one long token stream.

    Each item is (x,y) where y is x shifted right by one token - the
    self-supervision signal for next-token prediction. Windows overlap and
    start at every valid index, so shuffling the DataLoader mixes far more
    training signal out of the same text than chopping it into disjoint blocks.
    """

    def __init__(self,tokens,seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self,i):
        x = self.tokens[i:i+self.seq_len]
        y = self.tokens[i+1:i+self.seq_len+1]
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
    seq_len = 128
    batch_size = 128
    epochs = 3
    lr = 3e-4

    model_config = dict(emb_size=128,heads=8,num_layers=4,max_len=seq_len)
    model = Decoder(**model_config)

    text = download_dataset()
    tokens = torch.tensor(model.tokenizer.Encode(text))
    dataset = TextDataset(tokens,seq_len)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train(model,dataloader,optimizer,loss_fn,epochs,model.device)

    # save weights + the constructor args needed to rebuild the same
    # architecture, so inference doesn't have to guess or hardcode them
    os.makedirs(os.path.dirname(CHECKPOINT_PATH),exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model_config},CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")
