import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from penalty import penalty_method_dynamic_lr
from initial import initial
from prox_problem import energy_t
from utils import save_hyperparameters, visualize_loss, visualize_lr, visualize_energy

def generate_random_rho_weight():
    rho0 = torch.rand(1, 5)
    rho0 /= rho0.sum(dim=-1, keepdim=True)
    
    rho1 = torch.rand(1, 5)
    rho1 /= rho1.sum(dim=-1, keepdim=True)

    weight0 = torch.rand(1, 5, 5)
    weight0 = weight0 * (1 - torch.eye(5).unsqueeze(0))
    weight0 = weight0 + torch.eye(5).unsqueeze(0) * 1e-8
    weight0 = (weight0 + weight0.transpose(-1, -2)) / 2
    
    weight1 = torch.rand(1, 5, 5)
    weight1 = weight1 * (1 - torch.eye(5).unsqueeze(0))
    weight1 = weight1 + torch.eye(5).unsqueeze(0) * 1e-8
    weight1 = (weight1 + weight1.transpose(-1, -2)) / 2
    
    return rho0, rho1, weight0, weight1

def generate_random_b():
    b0 = torch.rand(1, 5, 5)
    b0 = b0 * (1 - torch.eye(5).unsqueeze(0))
    b0 = b0 + torch.eye(5).unsqueeze(0) * 1e-8
    b0 = (b0 - b0.transpose(-1, -2)) / 2
    
    b1 = torch.rand(1, 5, 5)
    b1 = b1 * (1 - torch.eye(5).unsqueeze(0))
    b1 = b1 + torch.eye(5).unsqueeze(0) * 1e-8
    b1 = (b1 - b1.transpose(-1, -2)) / 2
    
    return b0, b1

def main():
    # Experiment setup
    experiment_name = 'randomgraph_b'
    save_dir = os.path.join(experiment_name, 'output')
    log_file = os.path.join(experiment_name, 'log.txt')
    os.makedirs(save_dir, exist_ok=True)

    # Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    rho0, rho1, weight0, weight1 = generate_random_rho_weight()
    rho0 = rho0.to(device=device, dtype=dtype)
    rho1 = rho1.to(device=device, dtype=dtype)
    weight0 = weight0.to(device=device, dtype=dtype)
    # weight1 = weight1.to(device=device, dtype=dtype)
    weight1 = weight0
    b0, b1 = generate_random_b()
    b0 = b0.to(device=device, dtype=dtype)
    b1 = b1.to(device=device, dtype=dtype)

    # Parameters
    N = rho0.shape[1]
    T = 10
    beta = 1e-3
    sigma = 10000.0
    lr = 1e-4
    max_iters = 1000000
    log_every_iters = 100
    save_every_iters = 10000

    # Save hyperparameters
    hyperparams = {
        "N": N,
        "T": T,
        "beta": beta,
        "sigma": sigma,
        "learning_rate": lr,
        "max_iterations": max_iters
    }
    save_hyperparameters(log_file, hyperparams)

    # Initialize variables
    rho, b, weight = initial(rho0, rho1, weight0, weight1, b0, b1, T, beta)

    # Run the penalty method
    rho, b, weight, loss_history, lr_history = penalty_method_dynamic_lr(
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

    # Save energy loss evolution (log scale)
    visualize_loss(experiment_name, loss_history)

    # Save learning rate evolution (log scale)
    visualize_lr(experiment_name, lr_history)

    # Plot and save energy evolution
    E_t = energy_t(rho, b, weight)[0]
    visualize_energy(experiment_name, E_t)

if __name__ == '__main__':
    main()
