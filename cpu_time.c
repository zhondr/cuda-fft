/* ENEL428 Distributed Computing Assignment 2012
 * Author: Campbell Sinclair
 * Email: cls76@uclive.ac.nz
 * Date: 4 October 2012
 *
 * Uses fftw library to perform FFTs sequentially using the CPU.
 * Outputs the time taken for FFTs (varying the array length) to a text file.
 *
 * Compile using:
 * gcc cpu_time.c -lfftw3 -lm -o cpu_time
 * Run using:
 * ./cpu_time
 *
 * Based on:
 * http://marburg-md.googlecode.com/svn-history/r22/trunk/basti/stuff/better_dft.c
 */

#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265359

// This calculates the magnitude of the real&imaginary components, then overwrites the result into the x component memory (y memory no longer needed).
void CalcMagnitude(fftw_complex *data, long signalLength) {
    int i;
    for (i = 0; i < signalLength; i++) {
        data[i][0] = sqrt(data[i][0] * data[i][0] + data[i][1] * data[i][1]);
    }
}

// Finds the maximum magnitude (out of x array), and stores result in y[0].
void FindMaximum(fftw_complex *data, long signalLength) {
    // use data[0][1] as storage for maximum value
    data[0][1] = 0;
    
    int i;
    for (i = 0; i < signalLength; i++) {
        if (data[i][0] > data[0][1]) {
            data[0][1] = data[i][0];
        }
    }
}

// Normalise data by dividing values by maximum value.
void Normalise(fftw_complex *data, long signalLength) {
    int i;
    for (i = 0; i < signalLength; i++) {
        data[i][0] = data[i][0] / data[0][1];
    }
}

// Function called from main to do an FFT on a signal of length "signalLength".
// Returns time taken in microseconds
// (1) Generate set of data to be transformed
// (2) Do an FFT on the data
// (3) Normalise data
long transform (long signalLength) {
    fftw_complex *data;
    fftw_plan plan;
    struct timeval start, end;
    
    // Allocate data matrix
    data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * signalLength);
    
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
        data[i][0] = cos(2.0 * PI * signalFrequency * signalTime)
                      + cos(2.0 * PI * signalFrequency2 * signalTime)
                      + cos(2.0 * PI * signalFrequency3 * signalTime)
                      + (((double) (rand() % 2000000) - 1000000.0) / 2000000.0);
        data[i][1] = 0;
        
        signalTime += samplingPeriod;
    }
    
    // Plan for running the FFT
    plan = fftw_plan_dft_3d(signalLength, 1, 1, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // ---- START PERFORMANCE COMPARISON ----
    gettimeofday(&start, NULL);
    
    // (2) Do an FFT on the data
    fftw_execute(plan);
    
    // (3a) Normalise data: calculate magnitude
    CalcMagnitude(data, signalLength);
    // (3b) Normalise data: find maximum
    FindMaximum(data, signalLength);
    // (3c) Normalise data: divide all by maximum
    Normalise(data, signalLength);
    
    // ---- END PERFORMANCE COMPARISON ----
    gettimeofday(&end, NULL);
    
    // Clean up memory no longer needed
    fftw_destroy_plan(plan);
    fftw_free(data);
    
    return ((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
}

// Calculate the average for a set of data
// Input the set of data "data[]", and length of this data "dataLength"
// Outputs the average
double average(long data[], long dataLength) {
    double sum = 0;
    int i;
    for (i = 0; i < dataLength; i++) {
        sum += data[i];
    }
    
    return (sum / dataLength);
}


// -----------------
// Main program loop
// -----------------
int main(int argc, char** argv) {
    FILE *fp = fopen("CpuTimeData_total.txt", "w");
    long numIterations_small = 10;
    long numIterations_large = 1000;
    long signalLength, iteration, iterationResult[numIterations_large];
    
    // Perform FFT for a range of signal lengths
    for (signalLength = 1; signalLength <= 4096; signalLength++) {
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
        
        printf("%li\n", signalLength);
    }
    
    fclose(fp);
    
    return 0;
}
