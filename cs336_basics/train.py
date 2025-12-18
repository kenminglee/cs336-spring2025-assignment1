from dataclasses import dataclass
import os
from os.path import isfile
from typing import Literal
import typing
import random
import time

import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jaxtyping import Float, Int 
import tyro
import wandb
from tqdm import tqdm

from cs336_basics import ROOT_DIR
from cs336_basics.bpe.tokenization import Tokenizer
from cs336_basics.nn import TransformerLM, cross_entropy_loss


def tokenize_dataset(dataset_path:str, tokenizer: Tokenizer, tokenized_dataset_path: str, split_token:bytes=b"<|endoftext|>", chunk_size:int=4096):
    def find_document_boundaries():
        with open(dataset_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            pos = 0
            document_boundaries = [0]
            while pos < file_size:
                f.seek(pos)
                chunk = f.read(chunk_size)
                found_at = chunk.find(split_token)
                if found_at!=-1:
                    document_boundaries.append(pos+found_at+len(split_token)+1)
                    pos = document_boundaries[-1]
                else:
                    pos += chunk_size
        return document_boundaries


    doc_boundaries = find_document_boundaries()
    num_tokens = 0
    with open(dataset_path, "rb") as f:
        for index in tqdm(range(len(doc_boundaries)-1), desc="Counting num tokens in dataset"):
            f.seek(doc_boundaries[index])
            raw_bytes = f.read(doc_boundaries[index+1]-doc_boundaries[index])
            num_tokens += len(tokenizer.encode(raw_bytes.decode("utf-8", errors="ignore")))
    
    tokens_mm = np.memmap(
        tokenized_dataset_path,
        dtype=np.uint16,
        mode="w+",
        shape=(num_tokens,)
    )
    idx = 0
    with open(dataset_path, "rb") as f:
        for index in tqdm(range(len(doc_boundaries)-1), desc="tokenizing to file"):
            f.seek(doc_boundaries[index])
            raw_bytes = f.read(doc_boundaries[index+1]-doc_boundaries[index])
            tokenIDs = tokenizer.encode(raw_bytes.decode("utf-8", errors="ignore"))
            tokens_mm[idx:idx+len(tokenIDs)] = tokenIDs
            idx += len(tokenIDs)
    tokens_mm.flush()
    return tokens_mm

def load_data(
    dataset: Int[np.ndarray, "num_tokens"],  # noqa: F821
    batch_size: int, 
    context_length: int, 
    device: str,
    rng: np.random.Generator | None = None
) -> tuple[
    Int[torch.Tensor, "batch_size context_length"], 
    Int[torch.Tensor, "batch_size context_length"]
]:
    
    if rng is None:
        rng = np.random.default_rng()
    starting_indices = rng.choice(len(dataset)-context_length, size=batch_size)
    offsets = np.arange(context_length)
    data_in = torch.tensor(dataset[starting_indices[:,None]+offsets], device=device, dtype=torch.long)
    data_out = torch.tensor(dataset[starting_indices[:,None]+(offsets+1)], device=device, dtype=torch.long)
    return data_in, data_out

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None: 
    torch.save(
        {
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "iteration":iteration
        },
        out
    )

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

@dataclass
class Args:
    """
    Arguments for LM training
    """
    seed: int = 0
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    context_length: int = 256
    d_model : int = 512
    d_ff: int = 1344
    """Dimension of feed-forward MLP"""
    theta: int = 10000
    """RoPE theta parameter"""
    num_layers: int = 4
    """Number of prenorm-transformer block"""
    num_heads: int = 16
    """Number of transformer heads"""
    num_checkpoints: int = 5
    batch_size: int = 64
    training_steps: int = 1500
    """Number of training steps. Each training step involves batch_size x context_length number of tokens"""
    dataset: Literal["TinyStories", "OpenWebText"] = "TinyStories"
    wandb_project_name: str = "cs336-a1"
    """the wandb's project name"""
    run_name: str | None = None
    """Optional run name"""


def train():
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.run_name is None:
        run_name = f"{args.dataset}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    else:
        run_name = f"{args.run_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"

    wandb.init(
        project=args.wandb_project_name,
        sync_tensorboard=True,
        config=vars(args),
        group=args.dataset,
        name=run_name,
        save_code=False,
    )

    if args.dataset=="TinyStories":
        vocab_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train-vocab.json")
        merges_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train-merges.txt")
        train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train.txt")
        valid_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-valid.txt")
        tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-train.bin")
        tokenized_valid_dataset_path = os.path.join(ROOT_DIR, "../data/TinyStories-valid.bin")
    else:
        vocab_path = os.path.join(ROOT_DIR, "../data/owt-train-vocab.json")
        merges_path = os.path.join(ROOT_DIR, "../data/owt-train-merges.txt")
        train_dataset_path = os.path.join(ROOT_DIR, "../data/owt_train.txt")
        valid_dataset_path = os.path.join(ROOT_DIR, "../data/owt_valid.txt")
        tokenized_train_dataset_path = os.path.join(ROOT_DIR, "../data/owt-train.bin")
        tokenized_valid_dataset_path = os.path.join(ROOT_DIR, "../data/owt-valid.bin")

    assert os.path.isfile(vocab_path) and os.path.isfile(merges_path), "Vocab or merges file not found! Please run cs336_basics/bpe/train.py to train a BPE."
    assert os.path.isfile(train_dataset_path) and os.path.isfile(valid_dataset_path), "Training/Validation dataset not found! Please follow the instructions in Readme.md to download the datasets."

    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, ["<|endoftext|>"]
    )

    if not os.path.isfile(tokenized_train_dataset_path):
        print("Tokenized training dataset not found! tokenizing dataset now...")
        train_data = tokenize_dataset(train_dataset_path, tokenizer, tokenized_train_dataset_path)
    else:
        train_data = np.memmap(
            tokenized_train_dataset_path,
            dtype=np.uint16,
            mode="r"
        )
    
    if not os.path.isfile(tokenized_valid_dataset_path):
        print("Tokenized validation dataset not found! tokenizing dataset now...")
        validation_data = tokenize_dataset(valid_dataset_path, tokenizer, tokenized_valid_dataset_path)
    else:
        validation_data = np.memmap(
            tokenized_valid_dataset_path,
            dtype=np.uint16,
            mode="r"
        )
    

    checkpoint = args.training_steps // args.num_checkpoints
    for step in range(1, args.training_steps+1):
        pass
        



