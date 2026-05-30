# 🚀 LeetGPU Solutions & Progress Tracker

🔗 **Profile:**[lzyrapx on LeetGPU](https://leetgpu.com/lzyrapx) | 🎯 **Challenges:** [LeetGPU Challenges](https://leetgpu.com/challenges)

> **Progress Summary:** Actively conquering GPU programming challenges across multiple frameworks. Currently focusing heavily on **CUDA** and **PyTorch**, with ongoing explorations into modern compilers and languages like **Triton**, **Mojo**, and **TinyGrad**.

---

### 🧮 Matrix & Linear Algebra
*Core BLAS operations, matrix manipulation, and quantized variations.*

| Problems | CUDA | PyTorch | Triton | Mojo | TinyGrad | Cute DSL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [Batched Matrix Multiplication](cuda/Batched%20Matrix%20Multiplication) | ✅ | ✅ | | | | |
| [Dot Product](cuda/Dot%20Product) | ✅ | ✅ | | | | |
| [FP16 Batched Matrix Multiplication](cuda/FP16%20Batched%20Matrix%20Multiplication) | ✅ | | | | | |
| [FP16 Dot Product](cuda/FP16%20Dot%20Product) | ✅ | | | | | |
| [GEMM (FP16)](cuda/GEMM%20(FP16)) | ✅ | ✅ | | | | |
| [INT8 Quantized MatMul](cuda/Quantized%20Matrix%20Multiplication%20(INT8)) | ✅ | ✅ | | | | |
| [Matrix Addition](cuda/Matrix%20Addition) | ✅ | | | | | |
| [Matrix Copy](cuda/Matrix%20Copy) | ✅ | ✅ | | ✅ | | |
| [Matrix Multiplication](cuda/Matrix%20Multiplication) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Matrix Power](cuda/Matrix%20Power) | ✅ | | | | | |
| [Matrix Transpose](cuda/Matrix%20Transpose) | ✅ | ✅ | ✅ | ✅ | | ✅ |
| [Sparse Matrix-Vector Multiplication](cuda/Sparse%20Matrix-Vector%20Multiplication) | ✅ | ✅ | | | | |

### 🧠 Deep Learning & Neural Network Layers
*Attention mechanisms, normalizations, activations, and modern LLM kernels.*

| Problems | CUDA | PyTorch | Triton | Mojo | TinyGrad | Cute DSL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [Attention with Linear Biases](cuda/Attention%20with%20Linear%20Biases) | ✅ | | | | | |
| [Batch Normalization](cuda/Batch%20Normalization) | ✅ | | | | | |
| [Categorical Cross Entropy Loss](cuda/Categorical%20Cross%20Entropy%20Loss) | ✅ | ✅ | | | | |
| [Gaussian Error Gated Linear Unit](cuda/Gaussian%20Error%20Gated%20Linear%20Unit) | ✅ | | | | | |
| [Leaky ReLU](cuda/Leaky%20ReLU) | ✅ | ✅ | | ✅ | | |
| [Linear Self-Attention](cuda/Linear%20Self-Attention) | ✅ | | | | | |
| [LoRA Linear](cuda/LoRA%20Linear) | ✅ | | | | | |
| [Mean Squared Error](cuda/Mean%20Squared%20Error) | ✅ | ✅ | | | | |
| [Multi-Head Self-Attention](cuda/Multi-Head%20Self-Attention) | ✅ | | | | | |
| [ReLU](cuda/ReLU) | ✅ | ✅ | | ✅ | | |
| [RMS Normalization](cuda/RMS%20Normalization) | ✅ | | | | | |
| [Rotary Positional Embedding](cuda/Rotary%20Positional%20Embedding) | ✅ | | | | | |
|[Sigmoid Activation](cuda/Sigmoid%20Activation) | ✅ | | | | | |
|[Sigmoid Linear Unit](cuda/Sigmoid%20Linear%20Unit) | ✅ | | | | | |
|[Simple Inference](pytorch/Simple%20Inference) | | ✅ | | | | |
|[Sliding Window Self-Attention](cuda/Sliding%20Window%20Self-Attention) | ✅ | | | | | |
| [Softmax](cuda/Softmax) | ✅ | ✅ | | | | |
| [Softmax Attention](cuda/Softmax%20Attention) | ✅ | ✅ | | | | |
| [Swish-Gated Linear Unit](cuda/Swish-Gated%20Linear%20Unit) | ✅ | | | | | |
| [Weight Dequantization](cuda/Weight%20Dequantization) | ✅ | | | | | |

### 🖼️ Convolutions, Image & Signal Processing
*Filtering, FFT, max pooling, and spatial transformations.*

| Problems | CUDA | PyTorch | Triton | Mojo | TinyGrad | Cute DSL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [1D Convolution](cuda/1D%20Convolution) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [2D Convolution](cuda/2D%20Convolution) | ✅ | ✅ | | | | |
|[2D Max Pooling](cuda/2D%20Max%20Pooling) | ✅ | | | | | |
| [3D Convolution](cuda/3D%20Convolution) | ✅ | | | | | |
| [Color Inversion](cuda/Color%20Inversion) | ✅ | ✅ | | ✅ | | ✅ |
| [Fast Fourier Transform](cuda/Fast%20Fourier%20Transform) | ✅ | | | | | |
| [Gaussian Blur](cuda/Gaussian%20Blur) | ✅ | ✅ | | | | |
| [RGB to Grayscale](cuda/RGB%20to%20Grayscale) | ✅ | | | | | |

### 🧩 Core Algorithms, Memory & Arrays
*Parallel reductions, prefix sums, sorting, and array manipulations.*

| Problems | CUDA | PyTorch | Triton | Mojo | TinyGrad | Cute DSL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [2D Subarray Sum](cuda/2D%20Subarray%20Sum) | ✅ | | | | | |
|[3D Subarray Sum](cuda/3D%20Subarray%20Sum) | ✅ | | | | | |
| [Count Array Element](cuda/Count%20Array%20Element) | ✅ | ✅ | | | | |
| [Count 2D Array Element](cuda/Count%202D%20Array%20Element) | ✅ | ✅ | | | | |
| [Count 3D Array Element](cuda/Count%203D%20Array%20Element) | ✅ | | | | | |
|[Histogramming](cuda/Histogramming) | ✅ | ✅ | | | | |
| [Interleave Arrays](cuda/Interleave%20Arrays) | ✅ | | | | | |
|[Max Subarray Sum](cuda/Max%20Subarray%20Sum) | ✅ | | | | | |
| [Merge Sorted Arrays](cuda/Merge%20Sorted%20Arrays) | ✅ | | | | | |
| [Parallel Merge](cuda/Parallel%20Merge) | ✅ | | | | | |
| [Prefix Sum](cuda/Prefix%20Sum) | ✅ | ✅ | | | | |
| [Radix Sort](cuda/Radix%20Sort) | ✅ | ✅ | | | | |
| [Reduction](cuda/Reduction) | ✅ | ✅ | | | | |
| [Reverse Array](cuda/Reverse%20Array) | ✅ | ✅ | | ✅ | | |
| [Sorting](cuda/Sorting) | ✅ | ✅ | | | | |
| [Subarray Sum](cuda/Subarray%20Sum) | ✅ | | | | | |
| [Top-K Selection](cuda/Top-K%20Selection) | ✅ | ✅ | | | | |
| [Value Clipping](cuda/Value%20Clipping) | ✅ | | | | | |
| [Vector Addition](cuda/Vector%20Addition) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### ⚙️ Machine Learning, Graph & Others
*Stencils, regressions, graph traversal, and simulation algorithms.*

| Problems | CUDA | PyTorch | Triton | Mojo | TinyGrad | Cute DSL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [2D Jacobi Stencil](cuda/2D%20Jacobi%20Stencil) | ✅ | | | | | |
|[All-Pairs Shortest Paths](cuda/All-Pairs%20Shortest%20Paths) | ✅ | | | | | |
| [BFS Shortest Path](cuda/BFS%20Shortest%20Path) | ✅ | | | | | |
| [K-Means Clustering](cuda/K-Means%20Clustering) | ✅ | | | | | |
| [Linear Recurrence](cuda/Linear%20Recurrence) | ✅ | | | | | |
| [Logistic Regression](cuda/Logistic%20Regression) | ✅ | ✅ | | | | |
| [Monte Carlo Integration](cuda/Monte%20Carlo%20Integration) | ✅ | ✅ | | ✅ | | |
|[Multi-Agent Simulation](cuda/Multi-Agent%20Simulation) | ✅ | | | | | |
|[Nearest Neighbor](cuda/Nearest%20Neighbor) | ✅ | | | | | |
| [Ordinary Least Squares](cuda/Ordinary%20Least%20Squares%20Regression) | ✅ | ✅ | | | | |
| [Password Cracking](cuda/Password%20Cracking%20(FNV-1a)) | ✅ | | | | | |
| [Rainbow Table](cuda/Rainbow%20Table) | ✅ | ✅ | | ✅ | | |