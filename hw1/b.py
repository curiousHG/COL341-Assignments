import timeit
from matplotlib import pyplot as plt

times = []
high = 8
for k in range(high):
    k = 1
    N = 2**(k)
    mat = [[i+N-j for i in range(N)]for j in range(N)]

    ans = [[0 for _ in range(N)]for _ in range(N)]
    start = timeit.default_timer()

    for i in range(N):
        for j in range(N):
            for k in range(N):
                ans[i][j] += mat[i][k] * mat[k][j]

    stop = timeit.default_timer()
    times.append(stop-start)

plt.plot(range(high),times)
plt.yscale('log')
plt.show()

