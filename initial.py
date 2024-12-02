import torch
import numpy as np
from graph import degree, node_interpolation, grad, div
from utils import interpolant

def compute_init(tensor0: torch.Tensor, tensor1: torch.Tensor, T: int) -> torch.Tensor:
    """
    Interpolate between tensor0 and tensor1 over T time steps.
    
    Args:
        tensor0: (B, N, M, ...) Initial tensor.
        tensor1: (B, N, M, ...) Final tensor.
        T: int Number of time steps.
        
    Returns:
        tensor_interp: (B, T + 1, N, M, ...) Interpolated tensor.
    """
    # Get the shape of tensor0 (B, N, M, ...)
    shape = tensor0.shape

    # Create time_steps with shape (1, T, 1, 1, ...)
    time_steps = torch.linspace(0, 1, T + 1, device=tensor0.device)
    time_steps = time_steps.view(1, -1, *([1] * (len(shape) - 1)))  # shape becomes (1, T + 1, 1, 1, ...)

    # Add a time dimension to tensor0 and tensor1: (B, 1, N, M, ...)
    tensor0_expanded = tensor0.unsqueeze(1)  # Shape becomes (B, 1, N, M, ...)
    tensor1_expanded = tensor1.unsqueeze(1)  # Shape becomes (B, 1, N, M, ...)

    # Interpolate tensor0 and tensor1 along the time_steps dimension
    tensor_interp = (1 - time_steps) * tensor0_expanded + time_steps * tensor1_expanded
    
    return tensor_interp

def initial(rho0: torch.Tensor, rho1: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor, 
            b0: torch.Tensor, b1: torch.Tensor, T: int, beta: float, 
            rho_interp_way: str = 'mid', weight_interp_way: str = 'mid', b_interp_way: str = 'mid') -> tuple:
    """
    Compute the initial values of rho, b, and weight for the optimization problem.
    
    Args:
        rho0: (B, N) Initial rho.
        rho1: (B, N) Final rho.
        weight0: (B, N, N) Initial weight matrix.
        weight1: (B, N, N) Final weight matrix.
        b0: (B, N, N) Initial b.
        b1: (B, N, N) Final b.
        T: int Number of time steps.
        beta: float Regularization coefficient.
        rho_interp_way: str Interpolation method for rho ('left', 'right', 'mid').
        weight_interp_way: str Interpolation method for weight ('left', 'right', 'mid').
        b_interp_way: str Interpolation method for b ('left', 'right', 'mid').
        
    Returns:
        tuple: (rho: (B, T+1, N), b: (B, T+1, N, N), weight: (B, T+1, N, N))
    """
    # Interpolate rho and weight over T time steps
    rho = compute_init(rho0, rho1, T)
    weight = compute_init(weight0, weight1, T)

    # Compute time derivative of rho
    dt = 1.0 / T
    drhodt = (rho[:, 1:, :] - rho[:, :-1, :]) / dt  # (B, T, N)

    # Interpolate rho, weight, and b for the time steps
    rho_interp = interpolant(rho, rho_interp_way)
    weight_interp = interpolant(weight, weight_interp_way)

    # Solve for potential phi at each time step
    theta = node_interpolation(rho_interp, weight_interp)
    rho_grad = grad(rho_interp, weight_interp)
    b = drhodt - beta * div(rho_grad, weight_interp) / degree(weight_interp)
    A = -weight_interp * theta
    A_diag = -torch.diag_embed(torch.sum(A, dim=-1), dim1=-2, dim2=-1)
    A += A_diag

    # Solve the system of equations for phi
    A_pseudo_inv = torch.linalg.pinv(A)
    phi = torch.matmul(A_pseudo_inv, b.unsqueeze(-1))
    phi = phi.squeeze(-1)  # Remove the extra dimension
    b_res = grad(phi, weight_interp)

    # Interpolate b based on the chosen method
    b0 = b0.unsqueeze(1)
    b1 = b1.unsqueeze(1)

    if b_interp_way == 'left':
        b = torch.cat([b0, b_res[:, 1:], b1], dim=1)
    elif b_interp_way == 'right':
        b = torch.cat([b0, b_res[:, :-1], b1], dim=1)
    elif b_interp_way == 'mid':
        b = torch.zeros_like(b_res)
        b[:, 0:1, :, :] = b0
        for t in range(1, T):
            b[:, t, :, :] = b_res[:, t-1, :, :] * 2 - b[:, t-1, :, :]
        b = torch.cat([b, b1], dim=1)
    else:
        b = torch.cat([b0, b_res, b1], dim=1)

    return rho, b, weight
