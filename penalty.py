import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import os
from prox_problem import energy, constraint
from utils import save_graph_visualization, log_training


def penalty_method(
    rho_init, b_init, weight_init, beta, sigma, log_file, save_dir,
    max_iter=1000, lr=1e-3, log_every_iters=100, save_every_iters=500, scheduler_step=100
):
    """
    Optimizes the penalty method to solve energy minimization problem with constraints.

    Parameters:
        rho_init (Tensor): Initial density tensor.
        b_init (Tensor): Initial b tensor (boundary conditions).
        weight_init (Tensor): Weight tensor used in the energy calculation.
        beta (float): A regularization parameter for the constraint.
        sigma (float): Penalty factor for the constraint.
        log_file (str): Path to the log file for training progress.
        save_dir (str): Directory to save the graph visualizations.
        max_iter (int): Maximum number of iterations for optimization.
        lr (float): Learning rate for the optimizer.
        log_every_iters (int): Frequency of logging training progress.
        save_every_iters (int): Frequency of saving visualizations.
        scheduler_step (int): Frequency of scheduler step updates.

    Returns:
        rho (Tensor): Optimized density tensor.
        b (Tensor): Optimized b tensor.
        weight (Tensor): Final weight tensor.
        loss_history (list): History of loss values during training.
        lr_history (list): History of learning rates during training.
    """

    # Initialize variables
    log_rho = torch.log(rho_init.clamp(min=1e-8))[:, 1:-1]  # Avoid log(0) or log(negative)
    log_rho = log_rho.clone().detach().requires_grad_(True)  # Set log_rho as a trainable variable
    b_optimized = b_init[:, 1:-1].clone().detach().requires_grad_(True)  # Optimized b variable
    weight = weight_init  # Weight tensor remains unchanged in optimization

    # Create optimizer (Adam for log_rho and b_optimized)
    optimizer = torch.optim.Adam([log_rho, b_optimized], lr=lr)

    # Learning rate scheduler (reduce learning rate when loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    loss_history = []  # Store loss values
    lr_history = []  # Store learning rate values

    # Main optimization loop
    for iteration in tqdm(range(max_iter + 1), desc="Optimizing"):
        optimizer.zero_grad()  # Reset gradients for each iteration

        # Reconstruct rho and b tensors using the optimized values
        rho = torch.exp(log_rho)
        rho = torch.cat([rho_init[:, 0:1], rho, rho_init[:, -1:]], dim=1)  # Reattach boundary conditions
        b = torch.cat([b_init[:, 0:1], b_optimized, b_init[:, -1:]], dim=1)  # Same for b

        # Compute losses: energy and constraint
        loss_energy = torch.mean(energy(rho, b, weight))  # Energy loss
        loss_constraint = torch.mean(constraint(rho, b, weight, beta))  # Constraint violation loss
        losses = [loss_energy, loss_constraint]
        
        # Total loss, weighted by sigma for the constraint term
        loss = loss_energy + loss_constraint * sigma
        loss_history.append(loss.item())  # Record loss value

        # Backpropagation (compute gradients)
        loss.backward()

        # Optimizer step (update variables)
        optimizer.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record the learning rate for analysis
        lr_history.append(current_lr)

        # Scheduler step: update the learning rate if needed
        if iteration % scheduler_step == 0:
            scheduler.step(loss)

        # Log the training progress
        if iteration % log_every_iters == 0:
            log_training(log_file, iteration, max_iter, losses, loss, current_lr)

        # Save visualizations periodically
        if iteration % save_every_iters == 0:
            save_graph_visualization(save_dir, rho, b, iteration, is_directed=True)

    return rho, b, weight, loss_history, lr_history
