#!/usr/bin/env python

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


def main():
    os.makedirs("data/plots", exist_ok=True)

    plot_pdf_empirical()

    plot_histogram_empirical()

    plot_histogram("RMSDs")
    plot_histogram("Dihedrals")
    plot_histogram("Distances")

    plot_calibration_curve("RMSDs")
    plot_calibration_curve("Dihedrals")
    plot_calibration_curve("Distances")

    plot_rate_heatmap()

    plot_mse_heatmap()


def plot_pdf_empirical():
    pdf = torch.load("data/metrics/empirical.pt")["PDF"]
    bins1 = torch.load("data/cvs/RMSD_1_bins.pt")
    bins2 = torch.load("data/cvs/RMSD_2_bins.pt")

    fig, ax = plt.subplots(figsize=(2, 2.5), layout="constrained")
    pcm = ax.pcolormesh(
        bins1, bins2, pdf.T, norm=mpl.colors.LogNorm(1e-2, 1e3), rasterized=True
    )
    ax.set_xlabel("RMSD$_1$ (nm)")
    ax.set_ylabel("RMSD$_2$ (nm)")
    fig.colorbar(pcm, ax=ax, label=r"$\pi$ (nm$^{-2}$)", location="top")

    fig.savefig("data/plots/PDF Empirical.pdf")


def plot_histogram_empirical():
    hist = torch.load("data/metrics/empirical.pt")["Histogram"]
    bins1 = torch.load("data/cvs/RMSD_1_bins.pt")
    bins2 = torch.load("data/cvs/RMSD_2_bins.pt")

    fig, ax = plt.subplots(figsize=(2, 2.5), layout="constrained")
    pcm = ax.pcolormesh(bins1, bins2, hist.T, vmin=0, vmax=1, rasterized=True)
    ax.set_xlabel("RMSD$_1$ (nm)")
    ax.set_ylabel("RMSD$_2$ (nm)")
    fig.colorbar(pcm, ax=ax, label="$q$", location="top")

    fig.savefig("data/plots/Histogram Empirical.pdf")


def plot_histogram(features):
    scores = ["VCN", "EVCN"]
    lags = [5, 500, 50000]
    lag_labels = ["1 ns", "100 ns", "10 µs"]

    bins1 = torch.load("data/cvs/RMSD_1_bins.pt")
    bins2 = torch.load("data/cvs/RMSD_2_bins.pt")

    fig, axes = plt.subplots(
        3, 2, figsize=(4, 6), layout="constrained", sharex=True, sharey=True
    )

    for row, (lag, lag_label) in enumerate(zip(lags, lag_labels)):
        for col, score in enumerate(scores):
            data = load_data(features, lag, score)
            hist = torch.mean(torch.stack([d["Histogram"] for d in data], dim=0), dim=0)

            ax = axes[row, col]
            pcm = ax.pcolormesh(bins1, bins2, hist.T, vmin=0, vmax=1, rasterized=True)
            ax.set_xlabel("RMSD$_1$ (nm)")
            ax.set_ylabel(f"$\\tau$ = {lag_label}\nRMSD$_2$ (nm)")
            ax.label_outer()
    for col, score in enumerate(scores):
        axes[0, col].set_title(score)

    fig.colorbar(pcm, ax=axes, location="top", label="$q$")

    fig.savefig(f"data/plots/Histogram {features}.pdf")


def plot_calibration_curve(features):
    scores = ["VCN", "EVCN"]
    lags = [5, 500, 50000]
    lag_labels = ["1 ns", "100 ns", "10 µs"]

    fig, axes = plt.subplots(
        3, 2, figsize=(4, 5.5), layout="constrained", sharex=True, sharey=True
    )

    for row, (lag, lag_label) in enumerate(zip(lags, lag_labels)):
        for col, score in enumerate(scores):
            data = load_data(features, lag, score)
            q = torch.cat([d["Calibration Curve"][0] for d in data])
            q_ref = torch.cat([d["Calibration Curve"][1] for d in data])

            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], "k")
            ax.plot(q, q_ref, ".")
            ax.set_xlabel("Predicted $q$")
            ax.set_ylabel(f"$\\tau$ = {lag_label}\nEmpirical $q$")
            ax.label_outer()
    for col, score in enumerate(scores):
        axes[0, col].set_title(score)

    fig.savefig(f"data/plots/Calibration Curve {features}.pdf")


def plot_rate_heatmap():
    all_features = ["RMSDs", "Dihedrals", "Distances"]
    scores = ["VCN", "EVCN"]
    lags = [5, 500, 50000]
    lag_labels = ["1 ns", "100 ns", "10 µs"]
    two_lags = [(5, 500), (5, 50000), (500, 50000)]
    two_lag_labels = ["1 ns, 100 ns", "1 ns, 10 µs", "100 ns, 10 µs"]

    rate_empirical = torch.load("data/metrics/empirical.pt")["Rate"]["Empirical"]

    rows = []
    for features in all_features:
        for lag in lags:
            for score in scores:
                data = load_data(features, lag, score)
                row = []
                for rate_lags in lags + two_lags:
                    for rate_score in scores:
                        row.append([d["Rate"][rate_lags][rate_score] for d in data])
                row = torch.mean(torch.tensor(row) / rate_empirical, dim=1).tolist()
                rows.append(row)
    heatmap = torch.tensor(rows)

    ylabel = "Committor hyperparameters"
    xlabel = "Rate hyperparameters"

    yticklabels = []
    for features in all_features:
        for lag in lag_labels:
            for score in scores:
                yticklabels.append(f"{features}, {lag}, {score}")
    yticks = torch.arange(len(yticklabels)) + 0.5

    xticklabels = []
    for lag in lag_labels + two_lag_labels:
        for score in scores:
            xticklabels.append(f"{lag}, {score}")
    xticks = torch.arange(len(xticklabels)) + 0.5

    cbar_label = "Predicted transition rate / Empirical transition rate"

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    pcm = ax.pcolormesh(
        heatmap, norm=mpl.colors.LogNorm(vmin=0.1, vmax=10), cmap="coolwarm"
    )
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{heatmap[i, j].item():.1f}",
                ha="center",
                va="center",
            )

    fig.colorbar(pcm, ax=ax, location="top", label=cbar_label)

    fig.savefig("data/plots/Rate.pdf")


def plot_mse_heatmap():
    all_features = ["RMSDs", "Dihedrals", "Distances"]
    scores = ["VCN", "EVCN"]
    lags = [5, 500, 50000]
    lag_labels = ["1 ns", "100 ns", "10 µs"]
    two_lags = [(5, 500), (5, 50000), (500, 50000)]
    two_lag_labels = ["1 ns, 100 ns", "1 ns, 10 µs", "100 ns, 10 µs"]

    rows = []
    for features in all_features:
        for lag in lags:
            for score in scores:
                data = load_data(features, lag, score)
                row = []
                row.append([d["MSE"]["Empirical"] for d in data])
                for mse_lags in two_lags:
                    for mse_score in scores:
                        row.append([d["MSE"][mse_lags][mse_score] for d in data])
                row = torch.sqrt(torch.mean(torch.tensor(row), dim=1)).tolist()
                rows.append(row)
    heatmap = torch.tensor(rows) * 100

    ylabel = "Committor hyperparameters"
    xlabel = "MSE hyperparameters"

    yticklabels = []
    for features in all_features:
        for lag in lag_labels:
            for score in scores:
                yticklabels.append(f"{features}, {lag}, {score}")
    yticks = torch.arange(len(yticklabels)) + 0.5

    xticklabels = []
    xticklabels.append("Empirical")
    for lag in two_lag_labels:
        for score in scores:
            xticklabels.append(f"{lag}, {score}")
    xticks = torch.arange(len(xticklabels)) + 0.5

    cbar_label = "RMSE ($10^{-2}$)"

    fig, ax = plt.subplots(figsize=(6, 8), layout="constrained")

    pcm = ax.pcolormesh(heatmap, vmin=0, vmax=20)
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{heatmap[i, j].item():.1f}",
                c="w",
                ha="center",
                va="center",
            )

    fig.colorbar(pcm, ax=ax, location="top", label=cbar_label)

    fig.savefig("data/plots/MSE.pdf")


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


def load_best(features, lag, score, split, seed=None):
    data = []
    for step in make_save_steps(20000):
        path = f"data/metrics/{features}_{lag}_{score}_{split}"
        if seed is not None:
            path += f"_{seed}"
        data.append(torch.load(f"{path}/{step}.pt"))
    idx = torch.argmin(torch.tensor([d["Rate"][lag][score] for d in data])).item()
    return data[idx]


def load_data(features, lag, score):
    splits = [0, 1]
    seeds = [0, 1, 2, 3, 4]
    data = []
    for split in splits:
        for seed in seeds:
            data.append(load_best(features, lag, score, split, seed))
    return data


if __name__ == "__main__":
    main()
