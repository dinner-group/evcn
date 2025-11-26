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
    assert len(ca) == 35
    atom_pairs = ca[np.stack(np.triu_indices(len(ca), k=1), axis=-1)]
    assert atom_pairs.shape == (35 * (35 - 1) // 2, 2)
    return mdtraj.compute_distances(t, atom_pairs, periodic=False)


def compute_rmsds(topology, trajectory, reference):
    helix_1 = np.arange(2, 10)
    helix_2 = np.arange(13, 19)
    helix_3 = np.arange(21, 32)
    helix_12 = np.concatenate([helix_1, helix_2])
    helix_23 = np.concatenate([helix_2, helix_3])
    helix_123 = np.concatenate([helix_1, helix_2, helix_3])

    t = mdtraj.load(trajectory, top=topology)
    ref = mdtraj.load(reference)

    sel = t.top.select("name CA")
    ref_sel = ref.top.select("name CA")

    rmsd_12 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_12], ref_atom_indices=ref_sel[helix_12]
    )
    rmsd_23 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_23], ref_atom_indices=ref_sel[helix_23]
    )
    rmsd_123 = mdtraj.rmsd(
        t, ref, atom_indices=sel[helix_123], ref_atom_indices=ref_sel[helix_123]
    )
    return rmsd_12, rmsd_23, rmsd_123


def main():
    with open("data/files.json", "r") as f:
        data = json.load(f)
    topology = data["topology"]
    trajectories = data["trajectories"]
    reference = data["reference"]

    phi_psi = [compute_phi_psi(topology, trajectory) for trajectory in trajectories]
    phi, psi = [split(torch.tensor(np.array(x))) for x in zip(*phi_psi)]
    assert phi.shape[-1] == psi.shape[-1] == 35 - 1

    # histogram CVs
    rmsds = [
        compute_rmsds(topology, trajectory, reference) for trajectory in trajectories
    ]
    rmsd_12, rmsd_23, rmsd_123 = [split(torch.tensor(np.array(x))) for x in zip(*rmsds)]
    rmsd_12_bins = torch.linspace(0, 0.8, 80 + 1)
    rmsd_23_bins = torch.linspace(0, 0.8, 80 + 1)
    os.makedirs("data/cvs", exist_ok=True)
    torch.save(rmsd_12, "data/cvs/RMSD_12.pt")
    torch.save(rmsd_23, "data/cvs/RMSD_23.pt")
    torch.save(rmsd_12_bins, "data/cvs/RMSD_12_bins.pt")
    torch.save(rmsd_23_bins, "data/cvs/RMSD_23_bins.pt")

    # reactant/product states
    unfolded = (rmsd_12 >= 0.4) & (rmsd_23 >= 0.4) & (rmsd_123 >= 0.5)
    folded = (rmsd_12 <= 0.1) & (rmsd_23 <= 0.1) & (rmsd_123 <= 0.15)
    os.makedirs("data/states", exist_ok=True)
    torch.save(unfolded, "data/states/unfolded.pt")
    torch.save(folded, "data/states/folded.pt")

    # input features
    rmsds = torch.stack([rmsd_12, rmsd_23, rmsd_123], dim=-1)
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
