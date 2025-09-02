import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    input_int64 = input.to(torch.int64)
    sorted_tensor = torch.sort(input_int64)[0]
    # convert to uint32
    output.copy_(sorted_tensor.to(torch.uint32))  