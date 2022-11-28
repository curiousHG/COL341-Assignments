import numpy as np
import timeit
from matplotlib import pyplot as plt
times1= []
times2 = []
high =  9
for k in range(high):
    N = 2**(k)
    # N = k
    mat = [[0 for i in range(N)]for j in range(N)]
    a = np.array(mat)
    start = timeit.default_timer()
    b = np.matmul(a,a)
    stop = timeit.default_timer()
    times1.append((stop-start)*(10**9))

    ans = [[0 for _ in range(N)]for _ in range(N)]
    start = timeit.default_timer()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ans[i][j] += mat[i][k] * mat[k][j]
    stop = timeit.default_timer()
    times2.append((stop-start)*(10**9))

for i in range(high):
    times1[i] = 2*N**3/times1[i]
    times2[i] = 2*N**3/times2[i]

plt.plot(range(high),times1,label='numpy')
plt.plot(range(high),times2, label='for loops')
# plt.yscale('log')
plt.legend()
plt.show()