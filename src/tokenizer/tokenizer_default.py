from transformers import AutoTokenizer

from config.config import config

# turn markers. the model has to see where a user turn ends and an assistant
# turn begins, otherwise it never learns to stop and just keeps writing both
# sides of the conversation
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"


class Tokenizer():
    def __init__(self, tokenizer_model=config.tokenizer, chat_tokens=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        if chat_tokens:
            # added as real special tokens so each marker is ONE id rather than
            # being split into "<", "|", "user", ... by BPE
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [USER_TOKEN, ASSISTANT_TOKEN]}
            )

        self.user_id = self.tokenizer.convert_tokens_to_ids(USER_TOKEN)
        self.assistant_id = self.tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
        self.eos_id = self.tokenizer.eos_token_id

    def Encode(self, text:str):
        return self.tokenizer.encode(text)

    def Decode(self, input_ids, skip_special_tokens=True):
        return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
