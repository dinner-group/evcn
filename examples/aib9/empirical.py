#!/usr/bin/env python

import argparse
import os

import torch

from evcn import histogram_empirical, rate_empirical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reactant")
    parser.add_argument("product")
    parser.add_argument("cv_1")
    parser.add_argument("cv_2")
    parser.add_argument("cv_bins_1")
    parser.add_argument("cv_bins_2")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # make deterministic if seed is specified
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    reactant = torch.load(args.reactant)
    product = torch.load(args.product)
    cv_1 = torch.load(args.cv_1)
    cv_2 = torch.load(args.cv_2)
    cv_bins_1 = torch.load(args.cv_bins_1)
    cv_bins_2 = torch.load(args.cv_bins_2)

    cvs = torch.stack([cv_1, cv_2], dim=-1)
    bins = (cv_bins_1, cv_bins_2)

    data = {
        "PDF": torch.histogramdd(cvs.reshape(-1, 2), bins, density=True)[0],
        "Histogram": histogram_empirical(reactant, product, cvs, bins),
        "Rate": {"Empirical": rate_empirical(reactant, product)},
    }
    os.makedirs("data/metrics", exist_ok=True)
    torch.save(data, "data/metrics/empirical.pt")


if __name__ == "__main__":
    main()
