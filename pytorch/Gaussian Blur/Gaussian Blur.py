import torch
import torch.nn.functional as F
# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, 
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):  
    # Reshape inputs from flattened arrays to 2D matrices
    input_2d = input.view(input_rows, input_cols)
    kernel_2d = kernel.view(kernel_rows, kernel_cols)
    
    # Add batch and channel dimensions for F.conv2d
    input_4d = input_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    
    # Pad the input to handle boundary conditions (zero padding)
    padding_y = kernel_rows // 2
    padding_x = kernel_cols // 2
    
    # Perform convolution with zero padding
    result = F.conv2d(input_4d, kernel_4d, padding=(padding_y, padding_x))
    
    # Reshape the result back to 2D
    output_2d = result.squeeze(0).squeeze(0)
    
    # Copy the result to the output tensor
    output.copy_(output_2d.view(-1))

import torch
import torch.nn.functional as F
# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, 
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):  
    # Reshape inputs from flattened arrays to 2D matrices
    input_2d = input.view(input_rows, input_cols)
    kernel_2d = kernel.view(kernel_rows, kernel_cols)
    
    # Add batch and channel dimensions for F.conv2d
    input_4d = input_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    
    # Pad the input to handle boundary conditions (zero padding)
    padding_y = kernel_rows // 2
    padding_x = kernel_cols // 2
    
    # Perform convolution with zero padding
    result = F.conv2d(input_4d, kernel_4d, padding=(padding_y, padding_x))
    
    # Reshape the result back to 2D
    output_2d = result.squeeze(0).squeeze(0)
    
    # Copy the result to the output tensor
    output.copy_(output_2d.view(-1))

