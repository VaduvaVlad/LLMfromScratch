import argparse

import torch

from src.transformer.transformer import Decoder
from src.train.train import CHECKPOINT_PATH


def load_model(checkpoint_path=CHECKPOINT_PATH):
    # map_location="cpu" only affects how the checkpoint's tensors are read from
    # disk; Decoder(**config) still builds on config.device, and load_state_dict
    # copies each cpu tensor into the matching cuda parameter automatically
    checkpoint = torch.load(checkpoint_path,map_location="cpu")
    model = Decoder(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt",nargs="?",default="Hello:")
    parser.add_argument("--max_new_tokens",type=int,default=200)
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--top_k",type=int,default=50)
    args = parser.parse_args()

    model = load_model()
    text = model.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
