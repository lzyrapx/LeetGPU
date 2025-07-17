from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer, stack_allocation
from gpu import barrier
from gpu.memory import AddressSpace
from math import ceildiv

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD

fn reduction_sum_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32,
) -> None:
    var shared = stack_allocation[BLOCK_SIZE, dtype, address_space=AddressSpace.SHARED]()

    var local_thread_id = thread_idx.x
    var global_start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc: Float32 = 0.0
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = global_start_idx + local_thread_id + i * BLOCK_SIZE
        if Int32(global_idx) < N:
            acc += input[global_idx]

    shared[local_thread_id] = acc
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if local_thread_id < stride:
            shared[local_thread_id] += shared[local_thread_id + stride]
        barrier()
        stride //= 2

    if local_thread_id == 0:
        output[block_idx.x] = shared[0]

fn final_calculation_kernel(
    total_sum_ptr: UnsafePointer[Float32],
    result_ptr: UnsafePointer[Float32],
    a: Float32,
    b: Float32,
    n_samples: Int32
) -> None:
    if thread_idx.x == 0 and block_idx.x == 0:
        var total_sum = total_sum_ptr[0]
        var interval_width = b - a
        var n_samples_f32 = Float32(n_samples)
        result_ptr[0] = interval_width * (total_sum / n_samples_f32)

@export
def solve(y_samples: UnsafePointer[Float32], result: UnsafePointer[Float32], a: Float32, b: Float32, n_samples: Int32):
    if n_samples <= 0:
        return

    var ctx = DeviceContext()

    var current_N = n_samples
    var d_input = y_samples
    var temp_buffer = ctx.enqueue_create_buffer[dtype](0)

    while current_N > 1:
        var num_blocks = ceildiv(current_N, ELEMENTS_PER_BLOCK)
        temp_buffer = ctx.enqueue_create_buffer[dtype](Int(num_blocks))
        var d_output = temp_buffer.unsafe_ptr()

        ctx.enqueue_function[reduction_sum_kernel](
            d_input,
            d_output,
            current_N,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        
        d_input = d_output
        current_N = num_blocks

    var total_sum_ptr = d_input

    ctx.enqueue_function[final_calculation_kernel](
        total_sum_ptr,
        result,
        a,
        b,
        n_samples,
        grid_dim=1,
        block_dim=1
    )

    ctx.synchronize()