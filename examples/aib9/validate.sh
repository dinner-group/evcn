#!/usr/bin/env bash

args=(
    "data/states/left-handed.pt"
    "data/states/right-handed.pt"
    "data/cvs/xi_1.pt"
    "data/cvs/xi_2.pt"
    "data/cvs/xi_1_bins.pt"
    "data/cvs/xi_2_bins.pt"
)
for features in CVs Dihedrals; do
    for lag in 1 100 10000; do
        for score in VCN EVCN; do
            for seed in 0 1 2 3 4; do
                x="data/features/${features}.pt"

                input_dir="data/models/${features}_${lag}_${score}_0"
                output_dir="data/metrics/${features}_${lag}_${score}_0"
                python validate.py "${x}" "${args[@]}" --input_dir "${input_dir}" --output_dir "${output_dir}" --split 0.5 1.0 --seed "${seed}"

                input_dir="data/models/${features}_${lag}_${score}_1"
                output_dir="data/metrics/${features}_${lag}_${score}_1"
                python validate.py "${x}" "${args[@]}" --input_dir "${input_dir}" --output_dir "${output_dir}" --split 0.0 0.5 --seed "${seed}"
            done
        done
    done
done
