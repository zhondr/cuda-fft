import numpy as np
import matplotlib.pyplot as plt

data_file1 = np.loadtxt('CpuTimeData_onlyfft_AWS_c5.txt',delimiter='	')
data_file2 = np.loadtxt('GpuTimeData_onlyfft_AWS_p3.txt',delimiter='	')
data_file3 = np.loadtxt('FpgaTimeData_onlyfft.txt',delimiter='	')

fftsize = data_file1[:,0]
cputime = data_file1[:,1]
gputime = data_file2[:,1]
fpgatime = data_file3[:,1]

plt.plot(fftsize,cputime,fftsize,gputime,fftsize,fpgatime)
plt.legend(['CPU','GPU','FPGA'],loc='best')
# plt.plot(fftsize,cputime,fftsize,gputime)
# plt.legend(['CPU','GPU'],loc='best')
plt.xlabel('fftsize')
plt.ylabel('time in microseconds (µs)')
plt.show()

plt.semilogy(fftsize,cputime,fftsize,gputime,fftsize,fpgatime)
plt.legend(['CPU','GPU','FPGA'],loc='best')
# plt.semilogy(fftsize,cputime,fftsize,gputime)
# plt.legend(['CPU','GPU'],loc='best')
plt.xlabel('fftsize')
plt.ylabel('time in microseconds (µs)')
plt.show()
