#include <cuda_runtime.h>
#include <iostream>

/*
Kernel execution time: 1.1528 ms for 1024 * 1024 * 128 float elements
*/

// Error checking macro
#define CHECK_CUDA_ERROR(call)                                                 \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "      \
                      << "code: " << error << ", reason: "                     \
                      << cudaGetErrorString(error) << std::endl;               \
            exit(1);                                                           \
        }                                                                      \
    }

// Kernel for element-wise addition of float4 vectors
__global__ void addFloat4(const float4* a, const float4* b, float4* c, size_t n) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid].x = a[tid].x + b[tid].x;
        c[tid].y = a[tid].y + b[tid].y;
        c[tid].z = a[tid].z + b[tid].z;
        c[tid].w = a[tid].w + b[tid].w;
    }
}

// Function to initialize float4 arrays
void initializeFloat4(float4* vec, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        vec[i] = {1.0f, 2.0f, 3.0f, 4.0f};
    }
}

int main() {
    size_t numElements = 1024 * 1024 * 128/4;
    size_t size = numElements * sizeof(float4);

    // Allocate host memory
    float4* h_a = (float4*)malloc(size);
    float4* h_b = (float4*)malloc(size);
    float4* h_c = (float4*)malloc(size);

    if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
        std::cerr << "Failed to allocate host vectors." << std::endl;
        exit(1);
    }

    // Initialize host memory
    initializeFloat4(h_a, numElements);
    initializeFloat4(h_b, numElements);

    // Allocate device memory
    float4* d_a;
    float4* d_b;
    float4* d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, size));

    // Copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Launch the kernel and measure time
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

    addFloat4<<<gridSize, blockSize>>>(d_a, d_b, d_c, numElements); //warmup

    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    addFloat4<<<gridSize, blockSize>>>(d_a, d_b, d_c, numElements);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print some of the results and the execution time
    // for (size_t i = 0; i < 10; ++i) {
    //     std::cout << "h_c[" << i << "] = { "
    //               << h_c[i].x << ", " << h_c[i].y << ", " << h_c[i].z << ", "
    //               << h_c[i].w << " }" << std::endl;
    // }
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}