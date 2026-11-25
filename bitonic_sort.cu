
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// ==========================================
// CUDA ERROR CHECKING
// ==========================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(error)               \
                  << " at line " << __LINE__ << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// ==========================================
// BITONIC SORT CUDA KERNEL
// ==========================================

__global__ void bitonicSortKernel(
    int* data,
    int j,
    int k,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    int ixj = i ^ j;

    // Process each comparison pair only once
    if (ixj > i && ixj < n) {

        // Determine ascending or descending direction
        if ((i & k) == 0) {

            // Ascending order
            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }

        } else {

            // Descending order
            if (data[i] < data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// ==========================================
// CUSTOM GPU BITONIC SORT
// ==========================================

void bitonicSort(
    int* d_data,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Outer loop: sorting stages
    for (int k = 2; k <= n; k <<= 1) {

        // Inner loop: comparison steps
        for (int j = k >> 1; j > 0; j >>= 1) {

            bitonicSortKernel<<<blocks, threads, 0, stream>>>(
                d_data, j, k, n
            );

            CUDA_CHECK(cudaGetLastError());
        }
    }
}

// ==========================================
// CHECK WHETHER ARRAY IS SORTED
// ==========================================

bool verifySorted(const std::vector<int>& data) {
    return std::is_sorted(data.begin(), data.end());
}

// ==========================================
// CPU SORTING
// ==========================================

double cpuSort(
    const std::vector<int>& input,
    std::vector<int>& output
) {
    output = input;

    auto start = std::chrono::high_resolution_clock::now();

    std::sort(output.begin(), output.end());

    auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(
        stop - start
    ).count();
}

// ==========================================
// CUSTOM GPU SORTING WITH CUDA EVENTS
// ==========================================

void customGPUSort(
    const std::vector<int>& input,
    std::vector<int>& output,
    double& kernelTime,
    double& totalTime
) {
    int n = static_cast<int>(input.size());

    int* d_data = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_data,
        n * sizeof(int)
    ));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto totalStart = std::chrono::high_resolution_clock::now();

    // Copy input from CPU to GPU
    CUDA_CHECK(cudaMemcpyAsync(
        d_data,
        input.data(),
        n * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    ));

    // Measure only the custom sorting kernels
    CUDA_CHECK(cudaEventRecord(start, stream));

    bitonicSort(d_data, n, stream);

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed,
        start,
        stop
    ));

    kernelTime = elapsed;

    // Copy sorted result back to CPU
    output.resize(n);

    CUDA_CHECK(cudaMemcpyAsync(
        output.data(),
        d_data,
        n * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto totalStop = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration<double, std::milli>(
        totalStop - totalStart
    ).count();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

// ==========================================
// THRUST LIBRARY SORTING
// ==========================================

void thrustSort(
    const std::vector<int>& input,
    std::vector<int>& output,
    double& sortTime,
    double& totalTime
) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Copy input to GPU
    thrust::device_vector<int> d_data(
        input.begin(),
        input.end()
    );

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure Thrust sorting time
    CUDA_CHECK(cudaEventRecord(start));

    thrust::sort(d_data.begin(), d_data.end());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed,
        start,
        stop
    ));

    sortTime = elapsed;

    // Copy sorted data back to CPU
    output.resize(input.size());

    thrust::copy(
        d_data.begin(),
        d_data.end(),
        output.begin()
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    auto totalStop = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration<double, std::milli>(
        totalStop - totalStart
    ).count();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==========================================
// MAIN FUNCTION
// ==========================================

int main() {

    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "       PARALLEL SORTING ON THE GPU          \n";
    std::cout << "============================================\n";

    // Display GPU information
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "CUDA Capability: "
              << prop.major << "."
              << prop.minor << "\n";

    // Warm up CUDA runtime
    CUDA_CHECK(cudaFree(0));

    // Array sizes: 2^16 to 2^24
    const int powers[] = {
        16, 18, 20, 22, 24
    };

    std::mt19937 rng(42);

    std::uniform_int_distribution<int> dist(
        0, 1000000
    );

    std::cout << "\n";
    std::cout << "Generating random input data...\n";

    std::cout << "\n";
    std::cout << std::left
              << std::setw(12) << "Elements"
              << std::setw(14) << "CPU(ms)"
              << std::setw(16) << "Bitonic(ms)"
              << std::setw(16) << "Bitonic Total"
              << std::setw(14) << "Thrust(ms)"
              << std::setw(14) << "Thrust Total"
              << std::setw(14) << "Bitonic SP"
              << std::setw(14) << "Thrust SP"
              << "Verification\n";

    std::cout << std::string(120, '-') << "\n";

    // ======================================
    // TEST EACH ARRAY SIZE
    // ======================================

    for (int p : powers) {

        int n = 1 << p;

        std::cout << "\nTesting array size: "
                  << n << "\n";

        // Generate identical random input
        std::vector<int> input(n);

        for (int i = 0; i < n; i++) {
            input[i] = dist(rng);
        }

        std::vector<int> cpuOutput;
        std::vector<int> bitonicOutput;
        std::vector<int> thrustOutput;

        double cpuTime = 0;
        double bitonicKernelTime = 0;
        double bitonicTotalTime = 0;
        double thrustKernelTime = 0;
        double thrustTotalTime = 0;

        // CPU sorting
        cpuTime = cpuSort(input, cpuOutput);

        // Custom CUDA Bitonic Sort
        customGPUSort(
            input,
            bitonicOutput,
            bitonicKernelTime,
            bitonicTotalTime
        );

        // NVIDIA Thrust sorting
        thrustSort(
            input,
            thrustOutput,
            thrustKernelTime,
            thrustTotalTime
        );

        // Verify all results
        bool cpuValid = verifySorted(cpuOutput);
        bool bitonicValid = verifySorted(bitonicOutput);
        bool thrustValid = verifySorted(thrustOutput);

        // Compare outputs for correctness
        bool sameResults =
            (cpuOutput == bitonicOutput) &&
            (cpuOutput == thrustOutput);

        bool allValid =
            cpuValid && bitonicValid &&
            thrustValid && sameResults;

        // Calculate speedups using end-to-end times
        double bitonicSpeedup =
            cpuTime / bitonicTotalTime;

        double thrustSpeedup =
            cpuTime / thrustTotalTime;

        // Print results
        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(14) << std::fixed
                  << std::setprecision(3) << cpuTime
                  << std::setw(16) << bitonicKernelTime
                  << std::setw(16) << bitonicTotalTime
                  << std::setw(14) << thrustKernelTime
                  << std::setw(14) << thrustTotalTime
                  << std::setw(14) << bitonicSpeedup
                  << std::setw(14) << thrustSpeedup
                  << (allValid ? "PASS" : "FAIL")
                  << "\n";

        if (!allValid) {
            std::cerr << "ERROR: Sorting verification failed!\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "       ALL SORTING TESTS COMPLETED          \n";
    std::cout << "============================================\n";

    return 0;
}
