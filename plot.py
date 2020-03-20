import numpy as np
import matplotlib.pyplot as plt

data_file1 = np.loadtxt('CpuTimeData_total.txt',delimiter='	')
data_file2 = np.loadtxt('GpuTimeData_total.txt',delimiter='	')


matrixsize = data_file1[:,0]
gpu = data_file2[:,1]
cpu = data_file1[:,1]

print(matrixsize)

plt.plot(matrixsize,cpu,matrixsize,gpu)
plt.legend(['CPU','GPU'],loc='best')
plt.xlabel('r')
plt.ylabel('time')
plt.show()

plt.semilogy(matrixsize,cpu,matrixsize,gpu)
plt.legend(['CPU','GPU'],loc='best')
plt.xlabel('r')
plt.ylabel('time')
plt.show()
