import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import os
from prox_problem import energy, constraint
from utils import save_graph_visualization, log_training, save_graph_visualization_b

def penalty_method_dynamic_lr(
    rho_init, b_init, weight_init, beta, sigma, log_file, save_dir,
    max_iter=1000, lr=1e-3, log_every_iters=100, save_every_iters=500, scheduler_step=100
):
    # Initialize variables
    log_rho = torch.log(rho_init.clamp(min=1e-8))[:, 1:-1]  # Avoid log(0) or log(negative)
    log_rho = log_rho.clone().detach().requires_grad_(True)
    b_optim = b_init[:, 1:-1].clone().detach().requires_grad_(True)
    # log_weight = torch.log(weight_init.clamp(min=1e-8))[:, 1:-1]
    # log_weight = log_weight.clone().detach().requires_grad_(True)
    weight = weight_init

    # Create optimizer
    optimizer = torch.optim.Adam([log_rho, b_optim], lr=lr)
    
    # Create scheduler (Reduce learning rate when loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    loss_history = []
    lr_history = []

    for iteration in tqdm(range(max_iter + 1), desc="Optimizing"):
        optimizer.zero_grad()

        # Compute density tensor from log_rho
        rho = torch.exp(log_rho)
        rho = torch.cat([rho_init[:, 0:1], rho, rho_init[:, -1:]], dim=1)
        # weight = torch.exp(log_weight)
        # weight = torch.cat([weight_init[:, 0:1], weight, weight_init[:, -1:]], dim=1)
        b = torch.cat([b_init[:, 0:1], b_optim, b_init[:, -1:]], dim=1)

        # Compute losses
        loss_energy = energy(rho, b, weight)
        loss_constraint = torch.mean(torch.sum(constraint(rho, b, weight, beta)**2, dim=2), dim=1)
        loss = torch.mean(loss_energy + loss_constraint * sigma)
        losses = [torch.mean(loss_energy), torch.mean(loss_constraint)]
        loss_history.append(loss.item())

        # Backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        if iteration % scheduler_step == 0:
            scheduler.step(loss)

        # Logging
        if iteration % log_every_iters == 0:
            log_training(log_file, iteration, max_iter, losses, loss, current_lr)

        # Save results
        if iteration % save_every_iters == 0:
            save_graph_visualization_b(save_dir, rho, b, iteration)

    return rho, b, weight, loss_history, lr_history
