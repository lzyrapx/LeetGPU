import torch

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
      input = input.view(1,1,input_rows,input_cols)
      kernel = kernel.view(1,1,kernel_rows,kernel_cols)
      output[:] = torch.nn.functional.conv2d(input=input, weight=kernel).view_as(output)
