/* ENEL428 Distributed Computing Assignment 2012
 * Author: Campbell Sinclair
 * Email: cls76@uclive.ac.nz
 * Date: 4 October 2012
 *
 * Uses Nvidia Cuda API to perform FFTs in parallel using the GPU.
 * Outputs the time taken for FFTs (varying the array length) to a text file.
 *
 * Compile using:
 * nvcc gpu_time.cu -arch=sm_61 -lcufft -o gpu_time
 * Run using:
 * ./gpu_time
 *
 * Based on:
 * Dave van Leeuwen's notes.
 */

#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS_PER_BLOCK 1024
#define PI 3.14159265359

// First kernel to run on device.
// This calculates the magnitude of the real&imaginary components, then overwrites the result into the x component memory (y memory no longer needed).
__global__ void CalcMagnitude(cuDoubleComplex *data) {
    int myIndex = threadIdx.x + MAX_THREADS_PER_BLOCK*blockIdx.x;
    data[myIndex].x = sqrt(data[myIndex].x * data[myIndex].x + data[myIndex].y * data[myIndex].y);
}

// Second kernel to run on device.
// Single thread, performed sequentially.
// Finds the maximum magnitude (out of x array), and stores result in y[0].
__global__ void FindMaximum(cuDoubleComplex *data, long signalLength) {
    // use data[0].y as storage for maximum value
    data[0].y = 0;

    int i;
    for (i = 0; i < signalLength; i++) {
        if (data[i].x > data[0].y) {
            data[0].y = data[i].x;
        }
    }
}

// Third kernel to run on device.
// Normalise data by dividing values by maximum value.
__global__ void Normalise(cuDoubleComplex *data) {
    int myIndex = threadIdx.x + MAX_THREADS_PER_BLOCK*blockIdx.x;
    data[myIndex].x = data[myIndex].x / data[0].y;
}

// Function called from main to do an FFT on a signal of length "signalLength".
// Returns time taken in microseconds
// (1) Generate set of data to be transformed
// (2) Copy data from host to GPU
// (3) Do an FFT on the data
// (4) Normalise data
// (5) Copy data from GPU to host
long transform (long signalLength) {
    cuDoubleComplex *d_data, *h_data;
    cufftHandle plan;
    struct timeval start, end;

    // Multiple blocks and threads for parallel computation
    dim3 blocksParallel(ceil(((double)signalLength)/((double)MAX_THREADS_PER_BLOCK)), 1, 1);
    dim3 threadsParallel(MAX_THREADS_PER_BLOCK, 1, 1);
    // Single block and thread for sequential computation
    dim3 blocksSequential(1, 1, 1);
    dim3 threadsSequential(1, 1, 1);

    // Allocate host side matrix
    h_data = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * signalLength);
    // Allocate GPU side matrix
    cudaMalloc((void**) &d_data, sizeof(cuDoubleComplex) * signalLength);

    // (1) Generate set of data:
    // 3 sinusoids at 25, 50, 75 Hz + random noise.
    double samplingFrequency = 200.0;
    double samplingPeriod = 1.0/samplingFrequency;
    double signalFrequency = 25.0;
    double signalFrequency2 = 50.0;
    double signalFrequency3 = 75.0;
    double signalTime = 0.0;
    int i;
    for (i = 0; i < signalLength; i++) {
        h_data[i].x = cos(2.0 * PI * signalFrequency * signalTime)
                      + cos(2.0 * PI * signalFrequency2 * signalTime)
                      + cos(2.0 * PI * signalFrequency3 * signalTime)
                      + (((double) (rand() % 2000000) - 1000000.0) / 2000000.0);
        h_data[i].y = 0;

        signalTime += samplingPeriod;
    }

    // Plan for running the FFT
    cufftPlan3d(&plan, signalLength, 1, 1, CUFFT_Z2Z);

    // ---- START PERFORMANCE COMPARISON ----
    gettimeofday(&start, NULL);

    // (2) Copy data from host to GPU
    cudaMemcpy(d_data, h_data, sizeof(cuDoubleComplex)*signalLength, cudaMemcpyHostToDevice);

    // (3) Do an FFT on the data
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // (4a) Normalise data: calculate magnitude
    CalcMagnitude<<<blocksParallel, threadsParallel>>>(d_data);
    // Wait for all threads to finish
    cudaDeviceSynchronize();
    // (4b) Normalise data: find maximum
    FindMaximum<<<blocksSequential, threadsSequential>>>(d_data, signalLength);
    // Wait for all threads to finish (only one thread but can't hurt)
    cudaDeviceSynchronize();
    // (4c) Normalise data: divide all by maximum
    Normalise<<<blocksParallel, threadsParallel>>>(d_data);
    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // (5) Copy data from GPU to host
    cudaMemcpy(h_data, d_data, sizeof(cuDoubleComplex) * signalLength, cudaMemcpyDeviceToHost);

    // ---- END PERFORMANCE COMPARISON ----
    gettimeofday(&end, NULL);

    // Clean up memory no longer needed
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_data);

    return ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
}

// Calculate the average for a set of data
// Input the set of data "data[]", and length of this data "dataLength"
// Outputs the average
double average(long data[], long dataLength) {
    double sum = 0;
    int i;
    for (i = 0; i < dataLength; i++) {
        printf("iteration i - %li\n", data[i]);
        sum += data[i];
    }

    return (sum / dataLength);
}


// -----------------
// Main program loop
// -----------------
int main(int argc, char** argv) {
    FILE *fp = fopen("GpuTimeData_total.txt", "w");
    long numIterations_small = 50;
    long numIterations_large = 50;
    long sig, signalLength, iteration, iterationResult[numIterations_large];

    // Perform FFT for a range of signal lengths
    for (sig = 1; sig <= 11; sig++) {

        signalLength=pow(2,sig);
        printf("%li\n", signalLength);
        // Do more iterations for tests that take less time
        if (signalLength <= 64) {
            for (iteration = 0; iteration < numIterations_large; iteration++) {
                // Perform transform, and store result in array element
                iterationResult[iteration] = transform(signalLength);
            }
            // Average the results and output to a text file
            fprintf(fp, "%li\t%f\n", signalLength, average(iterationResult, numIterations_large));
        }
        // Do less iterations for tests that take longer
        else {
            for (iteration = 0; iteration < numIterations_small; iteration++) {
                // Perform transform, and store result in array element
                iterationResult[iteration] = transform(signalLength);
            }
            // Average the results and output to a text file
            fprintf(fp, "%li\t%f\n", signalLength, average(iterationResult, numIterations_small));
        }


    }

    fclose(fp);

    return 0;
}
