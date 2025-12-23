# CS336 Spring 2025 Assignment 1: Basics

Following Stanford's CS336 Spring 2025 Assignment 1 (handout [here](./cs336_spring2025_assignment1_basics.pdf)), this repository contains full training pipeline code for a GPT-like Transformer language model.

Aside from the test cases, [`find_chunk_boundaries`](./cs336_basics/bpe/pretokenization.py) and the [`gpt2_bytes_to_unicode`](./cs336_basics/bpe/serialization.py) functions (which were given), all code in this codebase was written by hand, without any use of AI.

This includes implementation of:
- Byte-Pair Tokenization (training + using it to tokenize text)
- All components of training a transformer:
    - Pre-norm transformer blocks (e.g., SwiGLU feed-forward networks, multi-head self-attention, RoPE encoding)
    - Embedding Layer
    - Cross-entropy loss
    - AdamW optimizer
    - Cosine-LR schedule
    - Gradient clipping
    - Logging to WandB
- Text generation components, like
    - Top-k/top-p sampling
    - Softmax temperature customization

Additionally, we include various notes (mainly on implementation tricks for specific components) and Jupyter notebooks on auxiliary experiments (e.g., experimenting with tokenizer and differing learning rates).

The training script located at `cs336_basics/train.py` is also compatible with WandB sweep.
We include several WandB sweep config files in the `hyperparam_tuning` folder.


## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training instructions

### Training a BPE Tokenizer

Customize `cs336_basics/bpe/train.py`, then run it.

### Training a LM

Run `cs336_basics/train.py`. Hyperparameters can be specified via CLI (do `python cs336_basics/train.py --help`)  to see all options.

### Running a hyperparameter sweep

First create a sweep with the appropriate `.yaml` file in `hyperparameter_tuning` folder. E.g.,

`wandb sweep --project "cs336-a1" --name "sweep-test" hyperparam_tuning/lr_sweep.yaml`

This command will return the sweep ID. Next time, all we have to do is to run `wandb agent <entity>/<project-name>/<sweep-id>`.

### Sample sbatch script
To run on SLURM-based HPCs, here's a template script that we can build from:
 
```sh
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

cd $HOME/cs336-spring2025-assignment1-new/
source .venv/bin/activate

# so that dataset is at $SLURM_TMPDIR
cp -R $HOME/cs336-spring2025-assignment1-new/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/cs336-spring2025-assignment1-new

for i in {0..7}; do
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 wandb agent <entity>/cs336-a1/<sweep_id> &
done

wait
```