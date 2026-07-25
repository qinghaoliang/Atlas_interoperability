"""
train.py
========

Training routines for FLECHA.

This module implements:
- local single-site training;
- FedAvg federated learning;
- compute-matched epoch calculation for single-site baselines.
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.FLECHA.model import ConnectomeRegressor4L


def train_local(
    model,
    loader,
    device,
    epochs,
    learning_rate=1e-3,
    weight_decay=1e-4,
):
    """Train a model locally on one client's data."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    model.train()

    for _ in range(epochs):
        for x, y in loader:
            # BatchNorm requires batch size > 1 during training.
            if x.shape[0] < 2:
                continue

            x = x.to(device)
            y = y.to(device).view(-1, 1)

            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()


def federated_fedavg(
    loaders,
    device,
    input_size,
    rounds=100,
    local_steps=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
    hidden_1=256,
    hidden_2=128,
    hidden_3=64,
    dropout=0.2,
):
    """
    FedAvg training.

    For each round:
    1. each client performs local minibatch updates;
    2. model parameters are averaged across clients;
    3. averaged global weights are broadcast back to clients.
    """
    models = [
        ConnectomeRegressor4L(
            input_size=input_size,
            n_l1=hidden_1,
            n_l2=hidden_2,
            n_l3=hidden_3,
            dropout=dropout,
        ).to(device)
        for _ in loaders
    ]

    optimizers = [
        torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        for model in models
    ]

    global_model = ConnectomeRegressor4L(
        input_size=input_size,
        n_l1=hidden_1,
        n_l2=hidden_2,
        n_l3=hidden_3,
        dropout=dropout,
    ).to(device)

    iterators = [iter(loader) for loader in loaders]
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(rounds), desc="FedAvg", leave=False):
        for _ in range(local_steps):
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                try:
                    x, y = next(iterators[i])
                except StopIteration:
                    iterators[i] = iter(loaders[i])
                    x, y = next(iterators[i])

                # BatchNorm requires batch size > 1 in train mode.
                if x.shape[0] < 2:
                    continue

                x = x.to(device)
                y = y.to(device).view(-1, 1)

                optimizer.zero_grad()
                loss_fn(model(x), y).backward()
                optimizer.step()

        # --------------------------------------------------------------
        # Average model parameters
        # --------------------------------------------------------------
        with torch.no_grad():
            global_state = global_model.state_dict()

            for key in global_state:
                if global_state[key].dtype.is_floating_point:
                    global_state[key].copy_(
                        sum(model.state_dict()[key] for model in models)
                        / len(models)
                    )
                else:
                    # Non-floating buffers, e.g. BatchNorm counters.
                    global_state[key].copy_(models[0].state_dict()[key])

            global_model.load_state_dict(global_state)

            for model in models:
                model.load_state_dict(global_model.state_dict())

    return global_model


def compute_matched_epochs(
    n_samples_site,
    n_clients,
    rounds=100,
    local_steps=10,
    batch_size=64,
    epochs_min=5,
    epochs_max=80,
):
    """
    Match approximate sample-gradient evaluations between FL and single-site
    training.

    FL samples processed approximately:

        rounds × local_steps × n_clients × batch_size

    Local samples processed approximately:

        epochs × n_samples_site
    """
    fl_samples = rounds * local_steps * n_clients * batch_size
    epochs = int(np.ceil(fl_samples / max(1, n_samples_site)))
    return int(np.clip(epochs, epochs_min, epochs_max))
