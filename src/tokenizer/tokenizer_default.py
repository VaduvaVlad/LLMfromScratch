from transformers import AutoTokenizer

from config.config import config

class Tokenizer():
    def __init__(self, tokenizer_model=config.tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def Encode(self, text:str):
        return self.tokenizer.encode(text)

    def Decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
