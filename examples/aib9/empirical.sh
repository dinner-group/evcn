#!/usr/bin/env bash

args=(
    "data/states/left-handed.pt"
    "data/states/right-handed.pt"
    "data/cvs/xi_1.pt"
    "data/cvs/xi_2.pt"
    "data/cvs/xi_1_bins.pt"
    "data/cvs/xi_2_bins.pt"
)
python empirical.py "${args[@]}"
