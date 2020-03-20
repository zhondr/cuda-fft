/* ENEL428 Distributed Computing Assignment 2012
 * Author: Campbell Sinclair
 * Email: cls76@uclive.ac.nz
 * Date: 4 October 2012
 *
 * Uses Nvidia Cuda API to perform FFTs in parallel using the GPU.
 * Outputs the data for a 4096-element FFT to a text file.
 *
 * Compile using:
 * nvcc gpu_signal.cu -arch=sm_61 -lcufft -o gpu_signal
 * Run using:
 * ./gpu_signal
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
// (1) Generate set of data to be transformed
// (2) Copy data from host to GPU
// (3) Do an FFT on the data
// (4) Normalise data
// (5) Copy data from GPU to host
void transform (long signalLength) {
    cuDoubleComplex *d_data, *h_data;
    cufftHandle plan;

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

    // (2) Copy data from host to GPU
    cudaMemcpy(d_data, h_data, sizeof(cuDoubleComplex)*signalLength, cudaMemcpyHostToDevice);

    // Plan for running the FFT
    cufftPlan3d(&plan, signalLength, 1, 1, CUFFT_Z2Z);

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

    // Output data to a text file
    FILE *fp = fopen("GpuSignalData_frequencydomain.txt", "w");
    for (i = 0; i < signalLength; i++) {
        fprintf(fp, "%f\n", h_data[i].x);
    }
    fclose(fp);

    // Clean up memory no longer needed
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_data);
}


// -----------------
// Main program loop
// -----------------
int main(int argc, char** argv) {
    transform(2048);

    return 0;
}
