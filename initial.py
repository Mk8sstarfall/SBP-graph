import torch
import numpy as np
from graph import degree, interpolation, grad

def compute_init(tensor0, tensor1, T):
    """Interpolate between tensor0 and tensor1 over T time steps."""
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

def initial(rho0, rho1, weight0, weight1, b0, b1, T, beta):
    rho = compute_init(rho0, rho1, T)
    weight = compute_init(weight0, weight1, T)

    # Compute time derivative of rho
    dt = 1.0 / T
    drhodt = (rho[:, 1:, :] - rho[:, :-1, :]) / dt # (B, T, N)

    rho_mid = (rho[:, 1:, :] + rho[:, :-1, :]) / 2
    weight_mid = (weight[:, 1:, :, :] + weight[:, :-1, :, :]) / 2

    # Solve for potential phi at each time step
    theta = interpolation(rho_mid, weight_mid)
    # log_rho_j = torch.log(rho_mid).unsqueeze(-2)
    # log_rho_i = torch.log(rho_mid).unsqueeze(-1)
    # b = drhodt + beta * torch.sum(weight_mid * (log_rho_i - log_rho_j) * theta, dim=-1) # (B, T, N)
    b = drhodt + beta * torch.sum(weight_mid * (rho_mid.unsqueeze(-1) - rho_mid.unsqueeze(-2)), dim=-1) / degree(weight_mid)
    A = -weight_mid * theta
    A_diag = -torch.diag_embed(torch.sum(A, dim=-1), dim1=-2, dim2=-1)
    A += A_diag

    phi = torch.linalg.solve(A, b.unsqueeze(-1))
    phi = phi.squeeze(-1)
    b_res = grad(phi, weight_mid)
    b = torch.zeros_like(b_res)
    b = torch.cat([b0.unsqueeze(1), b], dim=1)
    for t in range(1, T):
        b[:, t, :, :] = b_res[:, t-1, :, :] * 2 - b[:, t-1, :, :]
    b[:, T, :, :] = b1

    return rho, b, weight
