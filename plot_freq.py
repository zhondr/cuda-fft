import numpy as np
import matplotlib.pyplot as plt

data_file1 = np.loadtxt('CpuSignalData_frequencydomain.txt')
#data_file2 = np.loadtxt('GpuSignalData_frequencydomain.txt',delimiter=',')


matrixsize = [i for i in range(data_file1.size)]
y=matrixsize[:]
#gpu = data_file2[:,0]
cpu = data_file1[:]

plt.plot(matrixsize,cpu)
plt.legend(['freq'],loc='best')
plt.xlabel('freq')
plt.ylabel('magnitude')
plt.show()

# plt.semilogy(matrixsize,cpu,matrixsize,gpu)
# plt.legend(['CPU','GPU'],loc='best')
# plt.xlabel('N')
# plt.ylabel('time')
# plt.show()
