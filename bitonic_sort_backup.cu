
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace std;
using namespace std::chrono;

// Check CUDA errors
#define CUDA_CHECK(call)                                      \
{                                                             \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
        cerr << "CUDA Error: " << cudaGetErrorString(error)   \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}

// GPU Bitonic Sort Kernel
__global__ void bitonicSortKernel(int *data, unsigned int j,
                                  unsigned int k, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    unsigned int ixj = i ^ j;

    if (ixj > i && ixj < n)
    {
        bool ascending = ((i & k) == 0);

        if ((data[i] > data[ixj]) == ascending)
        {
            int temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// Verify whether array is sorted
bool isSorted(const vector<int>& arr)
{
    return is_sorted(arr.begin(), arr.end());
}

// CPU Sorting
double cpuSort(vector<int>& arr)
{
    auto start = high_resolution_clock::now();

    sort(arr.begin(), arr.end());

    auto stop = high_resolution_clock::now();

    return duration<double, milli>(stop - start).count();
}

// GPU Bitonic Sorting
double bitonicSort(int *d_data, unsigned int n,
                   float &kernelTime)
{
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto totalStart = high_resolution_clock::now();

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_data, nullptr, 0, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));

    // Bitonic sorting stages
    for (unsigned int k = 2; k <= n; k <<= 1)
    {
        for (unsigned int j = k >> 1; j > 0; j >>= 1)
        {
            bitonicSortKernel<<<blocks, threads>>>(d_data, j, k, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    auto totalStop = high_resolution_clock::now();

    return duration<double, milli>(totalStop - totalStart).count();
}

int main()
{
    cout << "\n========================================\n";
    cout << "   EXPERIMENT 8: PARALLEL GPU SORTING\n";
    cout << "========================================\n";

    ofstream csv("sorting_results.csv");

    if (!csv.is_open())
    {
        cerr << "Cannot create CSV file!" << endl;
        return 1;
    }

    csv << "Array_Size,CPU_ms,Bitonic_Kernel_ms,"
        << "Bitonic_Total_ms,Thrust_Sort_ms,"
        << "Thrust_Total_ms,Bitonic_Speedup,"
        << "Thrust_Speedup\n";

    // Random number generator
    mt19937 generator(42);
    uniform_int_distribution<int> distribution(0, 1000000);

    const int maxPower = 24;

    for (int power = 16; power <= maxPower; power++)
    {
        unsigned int n = 1u << power;

        cout << "\n----------------------------------------\n";
        cout << "Testing Array Size: 2^" << power
             << " = " << n << endl;

        vector<int> input(n);

        for (unsigned int i = 0; i < n; i++)
            input[i] = distribution(generator);

        vector<int> cpuData = input;
        vector<int> gpuResult(n);

        // Allocate GPU memory
        int *d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));

        // 1. CPU SORT
        double cpuTime = cpuSort(cpuData);

        cout << "CPU std::sort: "
             << cpuTime << " ms" << endl;

        cout << "CPU Sorted: "
             << (isSorted(cpuData) ? "YES" : "NO") << endl;

        // 2. GPU BITONIC SORT
        CUDA_CHECK(cudaMemcpy(d_data, input.data(),
                              n * sizeof(int),
                              cudaMemcpyHostToDevice));

        float bitonicKernelTime = 0;

        // Measure kernel execution time
        cudaEvent_t bitonicStart, bitonicStop;
        CUDA_CHECK(cudaEventCreate(&bitonicStart));
        CUDA_CHECK(cudaEventCreate(&bitonicStop));

        const int threads = 256;
        int blocks = (n + threads - 1) / threads;

        auto bitonicTotalStart = high_resolution_clock::now();

        CUDA_CHECK(cudaEventRecord(bitonicStart));

        for (unsigned int k = 2; k <= n; k <<= 1)
        {
            for (unsigned int j = k >> 1; j > 0; j >>= 1)
            {
                bitonicSortKernel<<<blocks, threads>>>(
                    d_data, j, k, n);

                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaEventRecord(bitonicStop));
        CUDA_CHECK(cudaEventSynchronize(bitonicStop));

        CUDA_CHECK(cudaEventElapsedTime(
            &bitonicKernelTime, bitonicStart, bitonicStop));

        CUDA_CHECK(cudaMemcpy(gpuResult.data(), d_data,
                              n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        auto bitonicTotalStop = high_resolution_clock::now();

        double bitonicTotalTime =
            duration<double, milli>(
                bitonicTotalStop - bitonicTotalStart).count();

        cout << "Bitonic Kernel: "
             << bitonicKernelTime << " ms" << endl;

        cout << "Bitonic Total: "
             << bitonicTotalTime << " ms" << endl;

        cout << "Bitonic Sorted: "
             << (isSorted(gpuResult) &&
                 gpuResult == cpuData ? "YES" : "NO") << endl;

        CUDA_CHECK(cudaEventDestroy(bitonicStart));
        CUDA_CHECK(cudaEventDestroy(bitonicStop));

        // 3. GPU THRUST SORT
        CUDA_CHECK(cudaMemcpy(d_data, input.data(),
                              n * sizeof(int),
                              cudaMemcpyHostToDevice));

        float thrustSortTime = 0;

        cudaEvent_t thrustStart, thrustStop;
        CUDA_CHECK(cudaEventCreate(&thrustStart));
        CUDA_CHECK(cudaEventCreate(&thrustStop));

        auto thrustTotalStart = high_resolution_clock::now();

        CUDA_CHECK(cudaEventRecord(thrustStart));

        thrust::device_ptr<int> dev_ptr(d_data);

        thrust::sort(dev_ptr, dev_ptr + n);

        CUDA_CHECK(cudaEventRecord(thrustStop));
        CUDA_CHECK(cudaEventSynchronize(thrustStop));

        CUDA_CHECK(cudaEventElapsedTime(
            &thrustSortTime, thrustStart, thrustStop));

        CUDA_CHECK(cudaMemcpy(gpuResult.data(), d_data,
                              n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        auto thrustTotalStop = high_resolution_clock::now();

        double thrustTotalTime =
            duration<double, milli>(
                thrustTotalStop - thrustTotalStart).count();

        cout << "Thrust Sort: "
             << thrustSortTime << " ms" << endl;

        cout << "Thrust Total: "
             << thrustTotalTime << " ms" << endl;

        cout << "Thrust Sorted: "
             << (isSorted(gpuResult) &&
                 gpuResult == cpuData ? "YES" : "NO") << endl;

        CUDA_CHECK(cudaEventDestroy(thrustStart));
        CUDA_CHECK(cudaEventDestroy(thrustStop));

        // Calculate speedup using end-to-end time
        double bitonicSpeedup = cpuTime / bitonicTotalTime;
        double thrustSpeedup = cpuTime / thrustTotalTime;

        cout << "\nSpeedup (CPU / Bitonic GPU): "
             << bitonicSpeedup << "x" << endl;

        cout << "Speedup (CPU / Thrust GPU): "
             << thrustSpeedup << "x" << endl;

        // Save results
        csv << n << ","
            << cpuTime << ","
            << bitonicKernelTime << ","
            << bitonicTotalTime << ","
            << thrustSortTime << ","
            << thrustTotalTime << ","
            << bitonicSpeedup << ","
            << thrustSpeedup << "\n";

        CUDA_CHECK(cudaFree(d_data));

        cout << "Results saved successfully." << endl;
    }

    csv.close();

    cout << "\n========================================\n";
    cout << "       EXPERIMENT COMPLETED!\n";
    cout << " Results saved in sorting_results.csv\n";
    cout << "========================================\n";

    return 0;
}
