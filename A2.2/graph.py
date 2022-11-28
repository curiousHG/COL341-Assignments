import matplotlib.pyplot as plt
import numpy as np

k = "loss"

loss_a = np.loadtxt(f'a/{k}.txt')
loss_b = np.loadtxt(f'b/{k}.txt')

plt.title(f'{k}_a')
plt.plot(loss_a, label='a')
plt.ylabel(f'{k}')

plt.xlabel('epoch')
plt.tight_layout()
plt.savefig(f'{k}_a.png')
plt.clf()

plt.title(f'{k}_b')
plt.plot(loss_b, label='b')
plt.ylabel(f'{k}')
plt.xlabel('epoch')
plt.tight_layout()
plt.savefig(f'{k}_b.png')
plt.clf()

