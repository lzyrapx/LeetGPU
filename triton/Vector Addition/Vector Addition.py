import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 转换原始指针为Triton指针类型
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    
    # 计算当前块的处理范围
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码防止越界访问
    mask = offsets < n_elements
    
    # 加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 计算并存储结果
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# a_ptr, b_ptr, c_ptr 是GPU内存的整型指针地址
def solve(a_ptr: int, b_ptr: int, c_ptr: int, N: int):    
    # 配置执行参数
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)  # 计算需要的块数量
    
    # 启动核函数
    vector_add_kernel[grid](a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE)