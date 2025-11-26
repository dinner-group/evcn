#!/usr/bin/env bash

a="data/states/left-handed.pt"
b="data/states/right-handed.pt"
for features in CVs Dihedrals; do
    for lag in 1 100 10000; do
        for score in VCN EVCN; do
            for seed in 0 1 2 3 4; do
                x="data/features/${features}.pt"

                output_dir="data/models/${features}_${lag}_${score}_0"
                python train.py "${x}" "${a}" "${b}" "${lag}" --score "${score}" --output_dir "${output_dir}" --split 0.0 0.5 --seed "${seed}"

                output_dir="data/models/${features}_${lag}_${score}_1"
                python train.py "${x}" "${a}" "${b}" "${lag}" --score "${score}" --output_dir "${output_dir}" --split 0.5 1.0 --seed "${seed}"
            done
        done
    done
done
