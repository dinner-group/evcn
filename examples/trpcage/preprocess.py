#!/usr/bin/env python

import json
import os

import mdtraj
import numpy as np
import torch


def compute_phi_psi(topology, trajectory):
    t = mdtraj.load(trajectory, top=topology)
    _, phi = mdtraj.compute_phi(t, periodic=False)
    _, psi = mdtraj.compute_psi(t, periodic=False)
    return phi, psi


def compute_distances(topology, trajectory):
    t = mdtraj.load(trajectory, top=topology)
    ca = t.top.select("name CA")
    assert len(ca) == 20
    atom_pairs = ca[np.stack(np.triu_indices(len(ca), k=1), axis=-1)]
    assert atom_pairs.shape == (20 * (20 - 1) // 2, 2)
    return mdtraj.compute_distances(t, atom_pairs, periodic=False)


def compute_rmsds(topology, trajectory, reference):
    helix_1 = np.arange(1, 9)
    helix_2 = np.arange(10, 14)
    helix_12 = np.concatenate([helix_1, helix_2])

    t = mdtraj.load(trajectory, top=topology)
    ref = mdtraj.load(reference)

    sel = t.top.select("name CA")
    ref_sel = ref.top.select("name CA")

    rmsd_1 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_1], ref_atom_indices=ref_sel[helix_1]
    )
    rmsd_2 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_2], ref_atom_indices=ref_sel[helix_2]
    )
    rmsd_12 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_12], ref_atom_indices=ref_sel[helix_12]
    )
    return rmsd_1, rmsd_2, rmsd_12


def main():
    with open("data/files.json", "r") as f:
        data = json.load(f)
    topology = data["topology"]
    trajectories = data["trajectories"]
    reference = data["reference"]

    phi_psi = [compute_phi_psi(topology, trajectory) for trajectory in trajectories]
    phi, psi = [split(torch.tensor(np.array(x))) for x in zip(*phi_psi)]
    assert phi.shape[-1] == psi.shape[-1] == 20 - 1

    # histogram CVs
    rmsds = [
        compute_rmsds(topology, trajectory, reference) for trajectory in trajectories
    ]
    rmsd_1, rmsd_2, rmsd_12 = [split(torch.tensor(np.array(x))) for x in zip(*rmsds)]
    rmsd_1_bins = torch.linspace(0, 0.51, 51 + 1)
    rmsd_2_bins = torch.linspace(0, 0.27, 54 + 1)
    os.makedirs("data/cvs", exist_ok=True)
    torch.save(rmsd_1, "data/cvs/RMSD_1.pt")
    torch.save(rmsd_2, "data/cvs/RMSD_2.pt")
    torch.save(rmsd_1_bins, "data/cvs/RMSD_1_bins.pt")
    torch.save(rmsd_2_bins, "data/cvs/RMSD_2_bins.pt")

    # reactant/product states
    unfolded = (rmsd_1 >= 0.35) & (rmsd_2 >= 0.2) & (rmsd_12 >= 0.4)
    folded = (rmsd_1 <= 0.05) & (rmsd_2 <= 0.05) & (rmsd_12 <= 0.15)
    os.makedirs("data/states", exist_ok=True)
    torch.save(unfolded, "data/states/unfolded.pt")
    torch.save(folded, "data/states/folded.pt")

    # input features
    rmsds = torch.stack([rmsd_1, rmsd_2, rmsd_12], dim=-1)
    dihedrals = torch.cat(
        [torch.cos(phi), torch.sin(phi), torch.cos(psi), torch.sin(psi)], dim=-1
    )
    distances = [compute_distances(topology, trajectory) for trajectory in trajectories]
    distances = split(torch.tensor(np.array(distances)))
    os.makedirs("data/features", exist_ok=True)
    torch.save(rmsds, "data/features/RMSDs.pt")
    torch.save(dihedrals, "data/features/Dihedrals.pt")
    torch.save(distances, "data/features/Distances.pt")


def split(x):
    n = x.shape[1] // 2
    return torch.cat([x[:, :n], x[:, -n:]])


if __name__ == "__main__":
    main()
