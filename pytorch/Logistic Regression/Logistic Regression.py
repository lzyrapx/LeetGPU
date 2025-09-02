import torch

def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    beta.zero_()
    max_iter = 500
    tol = 1e-8
    l2_reg = 1e-6
    
    for iteration in range(max_iter):
        z = X @ beta
        p = torch.sigmoid(z)
        W = p * (1 - p)
        W = torch.clamp(W, min=1e-8)
        
        gradient = X.t() @ (p - y) + l2_reg * beta
        
        XW = X * W.unsqueeze(1)
        hessian = X.t() @ XW + l2_reg * torch.eye(n_features, device=X.device, dtype=X.dtype)
        
        delta = torch.linalg.solve(hessian, gradient)   
        
        beta_new = beta - delta
        if torch.norm(beta_new - beta) < tol:
            beta.copy_(beta_new)
            break
        beta.copy_(beta_new)