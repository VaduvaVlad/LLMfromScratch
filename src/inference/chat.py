import argparse

import torch

from src.inference.generate import load_model
from src.tokenizer.tokenizer_default import USER_TOKEN, ASSISTANT_TOKEN


def chat(model,max_new_tokens=200,temperature=0.8,top_k=50):
    """Interactive loop.

    History is kept as raw token ids and grown in place, so the model sees the
    whole conversation so far - that is the only memory it has. Once history
    outgrows max_len, generate_tokens() crops to the most recent window, which
    is exactly when the model starts forgetting the start of the conversation.
    """
    tokenizer = model.tokenizer
    # stop as soon as the model tries to open a new user turn, otherwise it
    # happily writes your next message for you
    stop_ids = {tokenizer.eos_id, tokenizer.user_id}

    history = torch.empty((1,0),dtype=torch.long,device=model.device)

    print("chat started - ctrl+c or empty line to quit\n")
    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            break
        if not user_input:
            break

        turn = USER_TOKEN + user_input + ASSISTANT_TOKEN
        turn_ids = torch.tensor(
            [tokenizer.Encode(turn)],dtype=torch.long,device=model.device
        )
        history = torch.cat([history,turn_ids],dim=1)

        prompt_len = history.shape[1]
        history = model.generate_tokens(
            history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_ids=stop_ids,
        )

        # decode only what was newly generated, not the whole transcript
        reply = tokenizer.Decode(history[0,prompt_len:].tolist())
        print(f"bot: {reply.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens",type=int,default=200)
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--top_k",type=int,default=50)
    args = parser.parse_args()

    model = load_model()
    chat(model,args.max_new_tokens,args.temperature,args.top_k)
