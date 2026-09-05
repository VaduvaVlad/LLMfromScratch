import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer.transformer import Decoder
from src.train.pretrain import PRETRAIN_CHECKPOINT
from src.train.train import (CHECKPOINT_PATH, TextDataset, load_ultrachat,
                             make_optimizer, make_scheduler, train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained",default=PRETRAIN_CHECKPOINT)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--epochs",type=int,default=1)
    # 10x lower than pretraining: the weights already encode language, and a
    # large lr here would wash that out and leave only the chat formatting
    parser.add_argument("--lr",type=float,default=3e-5)
    parser.add_argument("--save_every",type=int,default=1000)
    args = parser.parse_args()

    if not os.path.exists(args.pretrained):
        raise SystemExit(f"no pretrained weights at {args.pretrained} - run "
                         f"'python -m src.train.pretrain' first")

    checkpoint = torch.load(args.pretrained,map_location="cpu")
    model_config = checkpoint["config"]
    model = Decoder(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    print(f"loaded pretrained weights from {args.pretrained} "
          f"(step {checkpoint.get('step','?')})",flush=True)

    seq_len = model_config["max_len"]
    tokens = load_ultrachat(model.tokenizer)
    dataset = TextDataset(tokens,seq_len,stride=seq_len)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,
                            num_workers=2,pin_memory=True,drop_last=True)

    total_steps = len(dataloader)*args.epochs
    optimizer = make_optimizer(model,args.lr)
    scheduler = make_scheduler(optimizer,warmup_steps=min(200,total_steps//50),
                               total_steps=total_steps)

    # stage 2 of 2: teach turn-taking and assistant style on top of weights
    # that already know how language and the world work
    train(model,dataloader,optimizer,nn.CrossEntropyLoss(),args.epochs,model.device,
          scheduler=scheduler,model_config=model_config,
          checkpoint_path=CHECKPOINT_PATH,save_every=args.save_every)
