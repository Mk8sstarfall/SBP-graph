import torch
import matplotlib.pyplot as plt
import os
from penalty import penalty_method
from initial import initial
from prox_problem import energy
from utils import save_hyperparameters, visualize_loss, visualize_lr, visualize_energy

def generate_random_rho() -> tuple:
    """
    Generate two random rho tensors with shape (1, 5), normalized along the last dimension.
    
    Returns:
        tuple: (rho0: (1, 5), rho1: (1, 5)) Random normalized rho tensors.
    """
    rho0 = torch.rand(1, 5)
    rho0 /= rho0.sum(dim=-1, keepdim=True)
    
    rho1 = torch.rand(1, 5)
    rho1 /= rho1.sum(dim=-1, keepdim=True)
    
    return rho0, rho1

def generate_random_weight() -> tuple:
    """
    Generate two random weight tensors with shape (1, 5, 5), symmetric and with a small diagonal perturbation.
    
    Returns:
        tuple: (weight0: (1, 5, 5), weight1: (1, 5, 5)) Random symmetric weight tensors.
    """
    weight0 = torch.rand(1, 5, 5)
    weight0 = weight0 * (1 - torch.eye(5).unsqueeze(0))  # Remove diagonal
    weight0 = weight0 + torch.eye(5).unsqueeze(0) * 1e-8  # Add small value to diagonal
    weight0 = (weight0 + weight0.transpose(-1, -2)) / 2  # Symmetrize the matrix
    
    weight1 = torch.rand(1, 5, 5)
    weight1 = weight1 * (1 - torch.eye(5).unsqueeze(0))
    weight1 = weight1 + torch.eye(5).unsqueeze(0) * 1e-8
    weight1 = (weight1 + weight1.transpose(-1, -2)) / 2
    
    return weight0, weight1

def generate_random_b() -> tuple:
    """
    Generate two random b tensors with shape (1, 5, 5), skew-symmetric and with a small diagonal perturbation.
    
    Returns:
        tuple: (b0: (1, 5, 5), b1: (1, 5, 5)) Random skew-symmetric b tensors.
    """
    b0 = torch.rand(1, 5, 5)
    b0 = b0 * (1 - torch.eye(5).unsqueeze(0))  # Remove diagonal
    b0 = b0 + torch.eye(5).unsqueeze(0) * 1e-8  # Add small value to diagonal
    b0 = (b0 - b0.transpose(-1, -2)) / 2  # Make it skew-symmetric
    
    b1 = torch.rand(1, 5, 5)
    b1 = b1 * (1 - torch.eye(5).unsqueeze(0))
    b1 = b1 + torch.eye(5).unsqueeze(0) * 1e-8
    b1 = (b1 - b1.transpose(-1, -2)) / 2
    
    return b0, b1

def main():
    """
    Main function to setup and run the penalty method optimization.
    
    This function sets up the experiment, initializes variables, runs the optimization, and visualizes results.
    """
    # Experiment setup
    experiment_name = 'randomgraph_b'
    save_dir = os.path.join(experiment_name, 'output')
    log_file = os.path.join(experiment_name, 'log.txt')
    os.makedirs(save_dir, exist_ok=True)

    # Device and dtype setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Generate random variables for the problem setup
    rho0, rho1 = generate_random_rho()
    rho0 = rho0.to(device=device, dtype=dtype)
    rho1 = rho1.to(device=device, dtype=dtype)
    
    weight0, weight1 = generate_random_weight()
    weight0 = weight0.to(device=device, dtype=dtype)
    weight1 = weight0  # Use the same weight for both initial and final
    
    b0, b1 = generate_random_b()
    b0 = b0.to(device=device, dtype=dtype)
    b1 = b1.to(device=device, dtype=dtype)

    # Set parameters for the experiment
    N = rho0.shape[1]
    T = 10
    beta = 1e-3
    sigma = 10000.0
    lr = 1e-4
    max_iters = 300000
    log_every_iters = 100
    save_every_iters = 10000

    # Save hyperparameters for reproducibility
    hyperparams = {
        "N": N,
        "T": T,
        "beta": beta,
        "sigma": sigma,
        "learning_rate": lr,
        "max_iterations": max_iters
    }
    save_hyperparameters(log_file, hyperparams)

    # Initialize the variables (rho, b, weight) using the initial function
    rho, b, weight = initial(rho0, rho1, weight0, weight1, b0, b1, T, beta)

    # Run the penalty method for optimization
    rho, b, weight, loss_history, lr_history = penalty_method(
        rho_init=rho,
        b_init=b,
        weight_init=weight,
        beta=beta,
        sigma=sigma,
        log_file=log_file,
        save_dir=save_dir,
        max_iter=max_iters,
        lr=lr,
        log_every_iters=log_every_iters,
        save_every_iters=save_every_iters,
    )

    # Visualize the loss evolution (log scale)
    visualize_loss(experiment_name, loss_history)

    # Visualize the learning rate evolution (log scale)
    visualize_lr(experiment_name, lr_history)

    # Plot and visualize energy evolution
    E_t = energy(rho, b, weight, return_mean=False)[0]
    visualize_energy(experiment_name, E_t)

if __name__ == '__main__':
    main()
