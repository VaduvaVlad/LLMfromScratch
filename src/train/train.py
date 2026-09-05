import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.transformer.transformer import Decoder
from src.tokenizer.tokenizer_default import USER_TOKEN, ASSISTANT_TOKEN

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "decoder.pt")
ULTRACHAT_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ultrachat_tokens_{n}.pt")


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


def load_ultrachat(tokenizer,num_conversations=None,cache_template=ULTRACHAT_CACHE,chunk=2000):
    """Tokenize N UltraChat conversations into one flat token stream.

    num_conversations=None uses the whole split. The count is part of the cache
    filename, otherwise changing it would silently reuse the wrong cache.
    """
    cache_path = cache_template.format(n=num_conversations or "all")

    if os.path.exists(cache_path):
        print(f"loading cached tokens from {cache_path}",flush=True)
        return torch.load(cache_path)

    import numpy as np
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft")
    if num_conversations is not None:
        dataset = dataset.select(range(min(num_conversations,len(dataset))))
    print(f"tokenizing {len(dataset):,} conversations...",flush=True)

    eos = tokenizer.tokenizer.eos_token

    # accumulate numpy arrays rather than one giant python list: 250M python
    # ints would cost several GB in object overhead alone
    pieces = []
    for start in range(0,len(dataset),chunk):
        batch = dataset[start:start+chunk]
        texts = [format_conversation(m) + eos for m in batch["messages"]]

        # batch encode: far faster than calling Encode() once per conversation
        for conversation in tokenizer.tokenizer(texts)["input_ids"]:
            pieces.append(np.asarray(conversation,dtype=np.int32))

        print(f"  {min(start+chunk,len(dataset)):,}/{len(dataset):,}",flush=True)

    tokens = torch.from_numpy(np.concatenate(pieces))

    os.makedirs(os.path.dirname(cache_path),exist_ok=True)
    torch.save(tokens,cache_path)
    print(f"cached {len(tokens):,} tokens to {cache_path}",flush=True)
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
        # tokens are stored narrow (int32 tensor, or uint16 memmap for the
        # pretraining corpus) to keep them on disk; embedding accepts int32 but
        # cross entropy targets must be int64, so widen both here
        if isinstance(x,np.ndarray):
            return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
        return x.long(), y.long()


def make_optimizer(model,lr,weight_decay=0.1):
    """AdamW with decay on weight matrices only.

    Weight decay on biases and LayerNorm gains hurts - they are meant to be
    free to sit anywhere, not pulled toward zero. betas=(0.9,0.95) rather than
    the (0.9,0.999) default is the standard choice for language models.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9,0.95),
    )


def make_scheduler(optimizer,warmup_steps,total_steps):
    """Linear warmup then cosine decay to zero.

    Warmup stops the first few steps - when the model is random and gradients
    are huge - from wrecking the weights. Cosine decay lets it settle into a
    minimum instead of bouncing around it at full learning rate.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step/max(1,warmup_steps)
        progress = (step - warmup_steps)/max(1,total_steps - warmup_steps)
        return 0.5*(1.0 + math.cos(math.pi*min(1.0,progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)


def save_checkpoint(model,model_config,path=CHECKPOINT_PATH):
    # weights + the constructor args needed to rebuild the same architecture,
    # so inference doesn't have to guess or hardcode them
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model_config},path)
    print(f"saved checkpoint to {path}",flush=True)


def save_resumable(model,model_config,optimizer,scheduler,step,path):
    """Full training state, not just weights.

    Resuming from weights alone restarts Adam's moment estimates from zero and
    the LR schedule from the top, which visibly dents the loss. An overnight
    run needs the optimizer and scheduler too.
    """
    os.makedirs(os.path.dirname(path),exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "config": model_config,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
    },tmp)
    # write then rename: a crash mid-save leaves the previous checkpoint intact
    os.replace(tmp,path)


def load_resumable(path,model,optimizer=None,scheduler=None,device="cuda:0"):
    checkpoint = torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint.get("step",0)


def train(model,dataloader,optimizer,loss_fn,epochs,device,
          scheduler=None,clip=1.0,amp=True,log_every=50,
          model_config=None,checkpoint_path=CHECKPOINT_PATH,
          save_every=None,start_step=0):
    model.train()
    global_step = start_step
    last_log = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for step,(x,y) in enumerate(dataloader):
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)

            # bf16 autocast: ~2x faster on a 4090 and halves activation memory.
            # bf16 has the same exponent range as fp32, so unlike fp16 it needs
            # no gradient scaler
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=amp):
                logits = model(x)                                    # (N,seq_len,vocab_size)
                loss = loss_fn(logits.reshape(-1,logits.size(-1)),y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                # one bad batch can produce a huge gradient and blow up the
                # weights; clipping bounds the step size
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                tok_per_s = (x.numel()*log_every)/max(1e-9,time.time()-last_log)
                eta = (len(dataloader)-step)*(time.time()-last_log)/max(1,log_every)/3600
                last_log = time.time()
                print(f"epoch {epoch} step {step}/{len(dataloader)} "
                      f"(global {global_step}): loss {loss.item():.4f} lr {lr:.2e} "
                      f"| {tok_per_s/1000:.0f}k tok/s eta {eta:.1f}h",flush=True)

            if save_every and global_step % save_every == 0 and model_config is not None:
                save_resumable(model,model_config,optimizer,scheduler,global_step,checkpoint_path)

        print(f"epoch {epoch} done: avg loss {total_loss/len(dataloader):.4f}",flush=True)

        # save every epoch, not just at the end: a multi-hour run that dies
        # part way should not lose everything
        if model_config is not None:
            save_resumable(model,model_config,optimizer,scheduler,global_step,checkpoint_path)
            print(f"saved checkpoint to {checkpoint_path}",flush=True)
    return global_step


if __name__ == "__main__":
    seq_len = 512
    batch_size = 24
    # 2 passes over the full 253M-token set beats more passes over a slice:
    # same compute, more unique data, far less memorisation
    epochs = 2
    lr = 3e-4
    num_conversations = None

    # gpt2-small scale. the 30M version reached loss 2.78 - it modelled english
    # and assistant formatting fine, but had nowhere to store meaning. the
    # bottleneck was capacity, not data, so this is the change that matters.
    # dropout 0 because 253M tokens against 124M params cannot overfit: at this
    # ratio dropout is just noise
    model_config = dict(emb_size=768,heads=12,num_layers=12,max_len=seq_len,dropout=0.0)
    model = Decoder(**model_config)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    tokens = load_ultrachat(model.tokenizer,num_conversations)
    dataset = TextDataset(tokens,seq_len,stride=seq_len)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,
                            num_workers=2,pin_memory=True,drop_last=True)

    total_steps = len(dataloader)*epochs
    optimizer = make_optimizer(model,lr)
    scheduler = make_scheduler(optimizer,warmup_steps=min(2000,total_steps//20),
                               total_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    train(model,dataloader,optimizer,loss_fn,epochs,model.device,
          scheduler=scheduler,model_config=model_config)
