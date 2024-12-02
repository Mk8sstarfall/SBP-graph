import numpy as np
import torch

def grad(phi, weight):
    # phi: (B, T, N)
    # weight: (B, T, N, N)
    # return: (B, T, N, N)
    phi_i = phi.unsqueeze(-1)
    phi_j = phi.unsqueeze(-2)
    return torch.sqrt(weight) * (phi_i - phi_j)

def div(m, weight):
    # m: (B, T, N, N)
    # weight: (B, T, N, N)
    # return: (B, T, N)
    return -torch.sum(m * torch.sqrt(weight), dim=-1)

def degree(weight):
    # weight: (B, T, N, N)
    # return: (B, T, N)
    degree = torch.sum(weight, dim=-1)
    return degree / torch.sum(degree, dim=-1, keepdim=True)

def interpolation(rho, weight, eps=1e-8):
    # rho: (B, T, N)
    # weight: (B, T, N, N)
    # return: (B, T, N, N)
    d = degree(weight)
    rho = rho / d
    rho_i = rho.unsqueeze(-1)
    rho_j = rho.unsqueeze(-2)
    theta = 0.5 * (rho_i + rho_j)
    return theta

def inner_product(v1, v2, rho, weight):
    # v1: (B, T, N, N)
    # v2: (B, T, N, N)
    # rho: (B, T, )
    # return: (B, T, 1)
    theta = interpolation(rho, weight)
    return torch.sum(v1 * v2 * theta, dim=(-2, -1)) / 2
