import tinygrad
from tinygrad.nn import Conv1d

# input, kernel, output are tensors on the GPU
def solve(input: tinygrad.Tensor, kernel: tinygrad.Tensor, output: tinygrad.Tensor, 
         input_size: int, kernel_size: int):
     
     input_reshape = input.reshape(1,1,input_size)
     kernel_reshape = kernel.reshape(1,1, kernel_size)

     conv = Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False)
     conv.weight = kernel_reshape

     convolved_output = conv(input_reshape)
     output.assign(convolved_output.reshape(-1))
     output.realize()