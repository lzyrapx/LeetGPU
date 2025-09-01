import torch

# X, y, beta are tensors on the GPU
def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.view(n_samples, n_features)

    XtX = X.T @ X
    Xty = X.T @ y
    beta[:] = torch.linalg.solve(XtX, Xty)