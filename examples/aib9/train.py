#!/usr/bin/env python

import argparse
import os

import torch
from adam_atan2_pytorch import AdamAtan2
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from evcn import CommittorOutput, mlp, training_step, vcn_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("reactant")
    parser.add_argument("product")
    parser.add_argument("lag", type=int)
    parser.add_argument("--batch_size", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--score", default="EVCN")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", type=float, nargs=2, default=[0.0, 1.0])
    args = parser.parse_args()

    features = torch.load(args.features)
    reactant = torch.load(args.reactant)
    product = torch.load(args.product)
    lag = args.lag
    batch_size = args.batch_size
    device = torch.device(args.device)
    hidden_layers = args.hidden_layers
    hidden_size = args.hidden_size
    lr = args.lr
    num_steps = args.num_steps
    output_dir = args.output_dir
    score = args.score.upper()
    seed = args.seed
    split = args.split

    if score not in ["VCN", "EVCN"]:
        raise ValueError(f"score must be VCN or EVCN (got {score})")
    exact = score == "EVCN"

    # make training deterministic if seed is specified
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    split = slice(int(len(features) * split[0]), int(len(features) * split[1]))
    features = features[split]
    reactant = reactant[split]
    product = product[split]

    dataset = vcn_dataset(features, reactant, product, lag, exact=exact)
    num_samples = batch_size * num_steps
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler)

    sizes = [features[0].shape[-1]] + hidden_layers * [hidden_size] + [1]
    model = mlp(*sizes, activation=nn.SiLU, final_activation=CommittorOutput).to(device)
    optimizer = AdamAtan2(model.parameters(), lr=lr)

    save_steps = make_save_steps(num_steps)
    step = 0
    for batch in dataloader:
        step += 1
        training_step(model, optimizer, batch, lag, device=device)
        if step in save_steps:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/{step}.pt")


def make_save_steps(n):
    out = []
    k = 0
    while True:
        for a in [1, 2, 5]:
            v = a * 10**k
            if v > n:
                return out
            out.append(v)
        k += 1


if __name__ == "__main__":
    main()
