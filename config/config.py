from dataclasses import dataclass

@dataclass
class config():
    tokenizer = "gpt2"
    device = "cuda:0"