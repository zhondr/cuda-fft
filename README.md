cuda-fft
========

Computes FFTs using a graphics card with CUDA support, and compares this with a CPU.

There are four different programs...

SET A, producing FFT outputs to confirm the FFT works:
 - cpu_signal.c performs an FFT using the CPU and outputs the result in a text file.
 - gpu_signal.cu performs an FFT using the GPU and outputs the result in a text file.

SET B, producing time outputs to measure performance:
 - cpu_time.c performs FFTs using the CPU on data ranging from 1-element to 4096-elements and outputs the total time for each FFT in a text file.
 - gpu_time.cu performs FFTs using the GPU on data ranging from 1-element to 4096-elements and outputs the total time for each FFT in a text file.
