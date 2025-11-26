#!/usr/bin/env bash

args=(
    "data/states/unfolded.pt"
    "data/states/folded.pt"
    "data/cvs/RMSD_12.pt"
    "data/cvs/RMSD_23.pt"
    "data/cvs/RMSD_12_bins.pt"
    "data/cvs/RMSD_23_bins.pt"
)
python empirical.py "${args[@]}"
