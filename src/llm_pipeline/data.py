import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def load_tinystories(split: str = "train[:50000]"):
    """Returns a list of story strings from the TinyStories dataset."""
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return [x["text"] for x in ds]


def train_bpe_tokenizer(texts, vocab_size: int, save_txt_path: str,
                         n_texts_for_training: int = 20000):
    """Trains a byte-level BPE tokenizer on a slice of `texts` and returns it,
    along with the id of the `<|endoftext|>` special token."""
    with open(save_txt_path, "w") as f:
        f.write("\n".join(texts[:n_texts_for_training]))

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[save_txt_path], vocab_size=vocab_size, min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    return tokenizer, eot_id


def pack_into_blocks(texts, tokenizer, eot_id: int, block_size: int) -> torch.Tensor:
    """Concatenates all texts (EOT-separated) into one token stream and chops
    it into fixed-length (block_size + 1) blocks. Returns a LongTensor of
    shape (n_blocks, block_size + 1)."""
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t).ids)
        all_ids.append(eot_id)
    n_blocks = len(all_ids) // (block_size + 1)
    return torch.tensor(
        all_ids[: n_blocks * (block_size + 1)], dtype=torch.long
    ).view(n_blocks, block_size + 1)
