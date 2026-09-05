import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.transformer.transformer import Decoder
from src.train.pretrain_data import prepare_fineweb
from src.train.train import (TextDataset, make_optimizer, make_scheduler,
                             train, load_resumable)

PRETRAIN_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..",
                                   "checkpoints", "pretrained.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens",type=int,default=2_000_000_000)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--seq_len",type=int,default=512)
    parser.add_argument("--lr",type=float,default=6e-4)
    parser.add_argument("--save_every",type=int,default=1000)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()

    # stage 1 of 2: learn language and world knowledge from broad web text.
    # chat data teaches format only - it has no density of facts to learn from,
    # which is why training on ultrachat alone produced fluent nonsense
    model_config = dict(emb_size=768,heads=12,num_layers=12,
                        max_len=args.seq_len,dropout=0.0)
    model = Decoder(**model_config)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}",flush=True)

    tokens = prepare_fineweb(model.tokenizer,args.target_tokens)
    dataset = TextDataset(tokens,args.seq_len,stride=args.seq_len)

    optimizer = make_optimizer(model,args.lr)

    start_step = 0
    if args.resume and os.path.exists(PRETRAIN_CHECKPOINT):
        start_step = load_resumable(PRETRAIN_CHECKPOINT,model,optimizer,device=model.device)
        print(f"resuming from step {start_step}",flush=True)

    # shuffle=False so resuming is exact: skip the samples already consumed.
    # fineweb documents arrive in arbitrary order already, so sequential
    # reading costs nothing here
    consumed = start_step*args.batch_size
    if consumed:
        dataset = Subset(dataset,range(consumed,len(dataset)))
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,
                            num_workers=2,pin_memory=True,drop_last=True)

    total_steps = start_step + len(dataloader)
    scheduler = make_scheduler(optimizer,warmup_steps=min(2000,total_steps//50),
                               total_steps=total_steps)
    if start_step:
        # fast-forward the schedule to where the previous run stopped
        for _ in range(start_step):
            scheduler.step()

    print(f"{len(dataloader):,} steps to go (total {total_steps:,})",flush=True)

    train(model,dataloader,optimizer,nn.CrossEntropyLoss(),1,model.device,
          scheduler=scheduler,model_config=model_config,
          checkpoint_path=PRETRAIN_CHECKPOINT,save_every=args.save_every,
          start_step=start_step)
