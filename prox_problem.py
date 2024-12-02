import torch
from graph import inner_product, grad, div, node_interpolation
from utils import interpolant

def energy(rho: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, 
           rho_interp_way: str = 'mid', b_interp_way: str = 'mid', 
           weight_interp_way: str = 'mid', return_mean: bool = True) -> torch.Tensor:
    """
    Calculate the energy based on rho, b, and weight.

    Parameters:
        rho (Tensor): Shape (B, T, N)
        b (Tensor): Shape (B, T, N, N)
        weight (Tensor): Shape (B, T, N, N)
        rho_interp_way (str): Interpolation method for rho (default 'mid')
        b_interp_way (str): Interpolation method for b (default 'mid')
        weight_interp_way (str): Interpolation method for weight (default 'mid')
        return_mean (bool): Whether to return the mean value or not (default True)
        
    Returns:
        Tensor: Energy value, shape (B, 1) if return_mean is True, else shape (B, T).
    """
    # Interpolate the inputs
    rho = interpolant(rho, rho_interp_way)
    weight = interpolant(weight, weight_interp_way)
    b = interpolant(b, b_interp_way)
    
    # Compute the energy
    energy_value = inner_product(b, b, rho, weight)
    if return_mean:
        return torch.mean(energy_value, dim=1)
    else:
        return energy_value

def constraint(rho: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, 
              beta: float, rho_interp_way: str = 'mid', 
              b_interp_way: str = 'mid', weight_interp_way: str = 'mid', 
              return_norm: bool = True) -> torch.Tensor:
    """
    Calculate the constraint based on rho, b, weight, and beta.

    Parameters:
        rho (Tensor): Shape (B, T, N)
        b (Tensor): Shape (B, T-1, N, N)
        weight (Tensor): Shape (B, T, N, N)
        beta (float): Constant multiplier for the gradient term
        rho_interp_way (str): Interpolation method for rho (default 'mid')
        b_interp_way (str): Interpolation method for b (default 'mid')
        weight_interp_way (str): Interpolation method for weight (default 'mid')
        return_norm (bool): Whether to return the norm of the constraint (default True)
        
    Returns:
        Tensor: Constraint value, shape (B, 1) if return_norm is True, else shape (B, T-1).
    """
    # Time step size (assuming T > 1)
    dt = 1.0 / (rho.shape[1] - 1)
    
    # Compute rho differences
    drho = rho[:, 1:, :] - rho[:, :-1, :]
    
    # Interpolate inputs
    rho = interpolant(rho, rho_interp_way)
    weight = interpolant(weight, weight_interp_way)
    b = interpolant(b, b_interp_way)
    
    # Compute the interpolation theta
    theta = node_interpolation(rho, weight)
    
    # Compute the flux term
    flux = theta * (b - beta * grad(torch.log(rho), weight))
    
    # Compute the constraint
    constraint_value = drho / dt + div(flux, weight)
    
    # Return the norm or the constraint itself
    if return_norm:
        return torch.mean(torch.sum(constraint_value**2, dim=-1), dim=1)
    else:
        return constraint_value
