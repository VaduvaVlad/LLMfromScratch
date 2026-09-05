import os
import time

import numpy as np

FINEWEB_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fineweb_edu_{n}.bin")


def _label(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:g}B"
    return f"{n/1_000_000:g}M"


def prepare_fineweb(tokenizer,target_tokens=2_000_000_000,
                    path_template=FINEWEB_BIN,batch_texts=1000):
    """Stream FineWeb-Edu, tokenize, and write one flat uint16 token file.

    Written straight to disk as a memmap rather than held in memory: 2B tokens
    is 4GB even as uint16, and a python list of them would be ~50GB.

    uint16 is safe only while the vocabulary fits in 65536 ids - asserted below.
    """
    assert len(tokenizer.tokenizer) < 65536, "vocab too large for uint16 storage"

    path = path_template.format(n=_label(target_tokens))
    if os.path.exists(path):
        tokens = np.memmap(path,dtype=np.uint16,mode="r")
        print(f"using cached {path} ({len(tokens):,} tokens)",flush=True)
        return tokens

    from datasets import load_dataset

    print(f"streaming fineweb-edu until {target_tokens:,} tokens...",flush=True)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",
                           split="train",streaming=True)

    eos_id = tokenizer.eos_id
    # pid in the temp name: two runs started at once would otherwise open the
    # same temp path and write over each other
    tmp = f"{path}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(path),exist_ok=True)

    written = 0
    texts = []
    with open(tmp,"wb") as out:
        for example in dataset:
            texts.append(example["text"])
            if len(texts) < batch_texts:
                continue

            # one eos after each document so the model learns where texts end
            ids = []
            for encoded in tokenizer.tokenizer(texts)["input_ids"]:
                ids.extend(encoded)
                ids.append(eos_id)
            texts = []

            np.asarray(ids,dtype=np.uint16).tofile(out)
            written += len(ids)
            if written % (50_000_000) < len(ids):
                print(f"  {written/1e6:.0f}M / {target_tokens/1e6:.0f}M tokens",flush=True)
            if written >= target_tokens:
                break

    # windows keeps a lock on a just-closed multi-GB file for a while (defender
    # scans it), so the rename has to be retried rather than assumed
    for attempt in range(20):
        try:
            os.replace(tmp,path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            if attempt == 0:
                print("  waiting for the OS to release the file...",flush=True)
            time.sleep(5)

    print(f"wrote {written:,} tokens to {path}",flush=True)
    return np.memmap(path,dtype=np.uint16,mode="r")
