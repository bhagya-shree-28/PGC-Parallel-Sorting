
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) \
             << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void bitonicSortKernel(
    int* data, unsigned int j,
    unsigned int k, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

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

void runBitonic(int* d_data, unsigned int n)
{
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (unsigned int k = 2; k <= n; k <<= 1)
    {
        for (unsigned int j = k >> 1; j > 0; j >>= 1)
        {
            bitonicSortKernel<<<blocks, threads>>>(
                d_data, j, k, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

bool verify(const vector<int>& actual,
            const vector<int>& expected)
{
    return is_sorted(actual.begin(), actual.end())
           && actual == expected;
}

double cpuBenchmark(const vector<int>& input, int repetitions)
{
    double total = 0;

    for (int r = 0; r < repetitions; r++)
    {
        vector<int> data = input;

        auto start = high_resolution_clock::now();
        sort(data.begin(), data.end());
        auto stop = high_resolution_clock::now();

        total += duration<double, milli>(stop - start).count();
    }

    return total / repetitions;
}

int main()
{
    const int repetitions = 5;

    cout << "\n============================================\n";
    cout << " EXPERIMENT 8: IMPROVED GPU SORT BENCHMARK\n";
    cout << "============================================\n";

    ofstream csv("improved_sorting_results.csv");

    if (!csv)
    {
        cerr << "Cannot create CSV file!" << endl;
        return 1;
    }

    csv << "Array_Size,CPU_Avg_ms,"
        << "Bitonic_Kernel_Avg_ms,"
        << "Bitonic_Total_Avg_ms,"
        << "Thrust_Kernel_Avg_ms,"
        << "Thrust_Total_Avg_ms,"
        << "Bitonic_Speedup,"
        << "Thrust_Speedup\n";

    mt19937 generator(42);
    uniform_int_distribution<int> distribution(0, 1000000);

    // GPU warm-up
    cout << "\nPerforming GPU warm-up...\n";

    int* warmup = nullptr;
    CUDA_CHECK(cudaMalloc(&warmup, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMemset(warmup, 0, 1024 * sizeof(int)));
    runBitonic(warmup, 1024);
    CUDA_CHECK(cudaFree(warmup));

    cout << "GPU warm-up completed.\n";

    for (int power = 16; power <= 24; power++)
    {
        unsigned int n = 1u << power;

        cout << "\n--------------------------------------------\n";
        cout << "Array size: 2^" << power << " = " << n << endl;

        vector<int> input(n);

        for (auto& value : input)
            value = distribution(generator);

        // CPU reference result
        vector<int> reference = input;
        sort(reference.begin(), reference.end());

        double cpuTime = cpuBenchmark(input, repetitions);

        cout << "CPU average: " << cpuTime << " ms\n";

        int* d_data = nullptr;

        CUDA_CHECK(cudaMalloc(
            &d_data, n * sizeof(int)));

        vector<int> result(n);

        // Bitonic warm-up for this size
        CUDA_CHECK(cudaMemcpy(
            d_data, input.data(), n * sizeof(int),
            cudaMemcpyHostToDevice));

        runBitonic(d_data, n);

        double bitonicKernelTotal = 0;
        double bitonicEndToEndTotal = 0;

        for (int r = 0; r < repetitions; r++)
        {
            cudaEvent_t start, stop;

            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            auto totalStart = high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_data, input.data(), n * sizeof(int),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaEventRecord(start));

            runBitonic(d_data, n);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float kernelMs = 0;

            CUDA_CHECK(cudaEventElapsedTime(
                &kernelMs, start, stop));

            CUDA_CHECK(cudaMemcpy(
                result.data(), d_data, n * sizeof(int),
                cudaMemcpyDeviceToHost));

            auto totalStop = high_resolution_clock::now();

            bitonicKernelTotal += kernelMs;

            bitonicEndToEndTotal +=
                duration<double, milli>(
                    totalStop - totalStart).count();

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            if (!verify(result, reference))
            {
                cerr << "Bitonic verification FAILED!\n";
                return 1;
            }
        }

        double bitonicKernelAvg =
            bitonicKernelTotal / repetitions;

        double bitonicTotalAvg =
            bitonicEndToEndTotal / repetitions;

        cout << "Bitonic kernel average: "
             << bitonicKernelAvg << " ms\n";

        cout << "Bitonic total average: "
             << bitonicTotalAvg << " ms\n";

        cout << "Bitonic verification: PASSED\n";

        // Thrust warm-up
        CUDA_CHECK(cudaMemcpy(
            d_data, input.data(), n * sizeof(int),
            cudaMemcpyHostToDevice));

        thrust::device_ptr<int> ptr(d_data);
        thrust::sort(ptr, ptr + n);

        double thrustKernelTotal = 0;
        double thrustEndToEndTotal = 0;

        for (int r = 0; r < repetitions; r++)
        {
            cudaEvent_t start, stop;

            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            auto totalStart = high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_data, input.data(), n * sizeof(int),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaEventRecord(start));

            thrust::sort(ptr, ptr + n);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float kernelMs = 0;

            CUDA_CHECK(cudaEventElapsedTime(
                &kernelMs, start, stop));

            CUDA_CHECK(cudaMemcpy(
                result.data(), d_data, n * sizeof(int),
                cudaMemcpyDeviceToHost));

            auto totalStop = high_resolution_clock::now();

            thrustKernelTotal += kernelMs;

            thrustEndToEndTotal +=
                duration<double, milli>(
                    totalStop - totalStart).count();

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            if (!verify(result, reference))
            {
                cerr << "Thrust verification FAILED!\n";
                return 1;
            }
        }

        double thrustKernelAvg =
            thrustKernelTotal / repetitions;

        double thrustTotalAvg =
            thrustEndToEndTotal / repetitions;

        cout << "Thrust kernel average: "
             << thrustKernelAvg << " ms\n";

        cout << "Thrust total average: "
             << thrustTotalAvg << " ms\n";

        cout << "Thrust verification: PASSED\n";

        double bitonicSpeedup = cpuTime / bitonicTotalAvg;
        double thrustSpeedup = cpuTime / thrustTotalAvg;

        cout << "Bitonic speedup: "
             << bitonicSpeedup << "x\n";

        cout << "Thrust speedup: "
             << thrustSpeedup << "x\n";

        csv << n << ","
            << cpuTime << ","
            << bitonicKernelAvg << ","
            << bitonicTotalAvg << ","
            << thrustKernelAvg << ","
            << thrustTotalAvg << ","
            << bitonicSpeedup << ","
            << thrustSpeedup << "\n";

        CUDA_CHECK(cudaFree(d_data));

        cout << "Results saved.\n";
    }

    csv.close();

    cout << "\n============================================\n";
    cout << " IMPROVED BENCHMARK COMPLETED!\n";
    cout << " File: improved_sorting_results.csv\n";
    cout << "============================================\n";

    return 0;
}
