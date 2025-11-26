#!/usr/bin/env python

import argparse
import os

import torch
from torch import nn

from evcn import (
    CommittorOutput,
    calibration_curve,
    histogram,
    mlp,
    mse_empirical,
    mse_vcn2,
    rate_vcn,
    rate_vcn2,
    transform,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("reactant")
    parser.add_argument("product")
    parser.add_argument("cv_1")
    parser.add_argument("cv_2")
    parser.add_argument("cv_bins_1")
    parser.add_argument("cv_bins_2")
    parser.add_argument("--batch_size", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--input_dir", default="./")
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", type=float, nargs=2, default=[0.0, 1.0])
    args = parser.parse_args()

    features = torch.load(args.features)
    reactant = torch.load(args.reactant)
    product = torch.load(args.product)
    cv_1 = torch.load(args.cv_1)
    cv_2 = torch.load(args.cv_2)
    cv_bins_1 = torch.load(args.cv_bins_1)
    cv_bins_2 = torch.load(args.cv_bins_2)
    batch_size = args.batch_size
    device = torch.device(args.device)
    hidden_layers = args.hidden_layers
    hidden_size = args.hidden_size
    input_dir = args.input_dir
    num_steps = args.num_steps
    output_dir = args.output_dir
    seed = args.seed
    split = args.split

    lags = [5, 500, 50000]
    two_lags = [(5, 500), (5, 50000), (500, 50000)]

    # make validation deterministic if seed is specified
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    split = slice(int(len(features) * split[0]), int(len(features) * split[1]))
    features = features[split]
    reactant = reactant[split]
    product = product[split]
    cv_1 = cv_1[split]
    cv_2 = cv_2[split]

    cvs = torch.stack([cv_1, cv_2], dim=-1)
    bins = (cv_bins_1, cv_bins_2)

    sizes = [features[0].shape[-1]] + hidden_layers * [hidden_size] + [1]
    model = mlp(*sizes, activation=nn.SiLU, final_activation=CommittorOutput).to(device)

    for step in make_save_steps(num_steps):
        model.load_state_dict(torch.load(f"{input_dir}/{step}.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            committor = transform(
                model, features, reactant, product, batch_size=batch_size, device=device
            )

            data = {}

            data["Histogram"] = histogram(committor, cvs, bins)

            data["Calibration Curve"] = calibration_curve(
                committor, reactant, product, 20
            )

            data["Rate"] = {}
            for lag in lags:
                data["Rate"][lag] = {}
                data["Rate"][lag]["VCN"] = rate_vcn(
                    committor, reactant, product, lag, exact=False
                )
                data["Rate"][lag]["EVCN"] = rate_vcn(
                    committor, reactant, product, lag, exact=True
                )
            for lag1, lag2 in two_lags:
                data["Rate"][lag1, lag2] = {}
                data["Rate"][lag1, lag2]["VCN"] = rate_vcn2(
                    committor, reactant, product, lag1, lag2, exact=False
                )
                data["Rate"][lag1, lag2]["EVCN"] = rate_vcn2(
                    committor, reactant, product, lag1, lag2, exact=True
                )

            data["MSE"] = {}
            data["MSE"]["Empirical"] = mse_empirical(committor, reactant, product)
            for lag1, lag2 in two_lags:
                data["MSE"][lag1, lag2] = {}
                data["MSE"][lag1, lag2]["VCN"] = mse_vcn2(
                    committor, reactant, product, lag1, lag2, exact=False
                )
                data["MSE"][lag1, lag2]["EVCN"] = mse_vcn2(
                    committor, reactant, product, lag1, lag2, exact=True
                )

            os.makedirs(output_dir, exist_ok=True)
            torch.save(data, f"{output_dir}/{step}.pt")


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
