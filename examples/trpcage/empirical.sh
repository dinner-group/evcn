#!/usr/bin/env bash

args=(
    "data/states/unfolded.pt"
    "data/states/folded.pt"
    "data/cvs/RMSD_1.pt"
    "data/cvs/RMSD_2.pt"
    "data/cvs/RMSD_1_bins.pt"
    "data/cvs/RMSD_2_bins.pt"
)
python empirical.py "${args[@]}"
