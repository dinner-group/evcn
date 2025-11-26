#!/usr/bin/env python

import json
import os

import mdtraj
import numpy as np
import torch
from joblib import Parallel, delayed


def compute_phi_psi(topology, trajectory):
    phi_idx = 13 * np.arange(2, 7)[:, np.newaxis] + np.array([4, 6, 8, 17])
    psi_idx = 13 * np.arange(2, 7)[:, np.newaxis] + np.array([6, 8, 17, 19])

    t = mdtraj.load(trajectory, top=topology)
    phi = mdtraj.compute_dihedrals(t, phi_idx)
    psi = mdtraj.compute_dihedrals(t, psi_idx)
    return phi, psi


def main():
    with open("data/files.json", "r") as f:
        data = json.load(f)
    topology = data["topology"]
    trajectories = data["trajectories"]

    phi_psi = Parallel(n_jobs=-1)(
        delayed(compute_phi_psi)(topology, trajectory) for trajectory in trajectories
    )
    phi, psi = [torch.tensor(np.array(x)) for x in zip(*phi_psi)]

    # histogram CVs
    gamma = -0.8 * (torch.sin(phi) + torch.sin(psi))
    xi_1 = gamma @ torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    xi_2 = gamma @ torch.tensor([1.0, 1.0, 0.0, -1.0, -1.0])
    xi_1_bins = torch.linspace(-7.5, 7.5, 150 + 1)
    xi_2_bins = torch.linspace(-6, 6, 120 + 1)
    os.makedirs("data/cvs", exist_ok=True)
    torch.save(xi_1, "data/cvs/xi_1.pt")
    torch.save(xi_2, "data/cvs/xi_2.pt")
    torch.save(xi_1_bins, "data/cvs/xi_1_bins.pt")
    torch.save(xi_2_bins, "data/cvs/xi_2_bins.pt")

    # reactant/product states
    phi_deg = torch.rad2deg(phi)
    psi_deg = torch.rad2deg(psi)
    in_l = (phi_deg - 41) ** 2 + (psi_deg - 47) ** 2 <= 25**2
    in_r = (phi_deg + 41) ** 2 + (psi_deg + 47) ** 2 <= 25**2
    in_lllll = torch.all(in_l, dim=-1)
    in_rrrrr = torch.all(in_r, dim=-1)
    os.makedirs("data/states", exist_ok=True)
    torch.save(in_lllll, "data/states/left-handed.pt")
    torch.save(in_rrrrr, "data/states/right-handed.pt")

    # input features
    cvs = torch.stack([xi_1, xi_2], dim=-1)
    dihedrals = torch.cat(
        [torch.cos(phi), torch.sin(phi), torch.cos(psi), torch.sin(psi)], dim=-1
    )
    os.makedirs("data/features", exist_ok=True)
    torch.save(cvs, "data/features/CVs.pt")
    torch.save(dihedrals, "data/features/Dihedrals.pt")


if __name__ == "__main__":
    main()
