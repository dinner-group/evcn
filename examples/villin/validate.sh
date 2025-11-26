#!/usr/bin/env bash

args=(
    "data/states/unfolded.pt"
    "data/states/folded.pt"
    "data/cvs/RMSD_12.pt"
    "data/cvs/RMSD_23.pt"
    "data/cvs/RMSD_12_bins.pt"
    "data/cvs/RMSD_23_bins.pt"
)
for features in RMSDs Dihedrals Distances; do
    for lag in 5 500 50000; do
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
