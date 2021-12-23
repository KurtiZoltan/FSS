import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

data = np.loadtxt("data.txt", )
E = data[:, 0]
before = data[:, 1] * 0
beforeEs = []
for i in range(1, 6):
    before += data[:, i]
    avg = np.sum(E * data[:, i]) / np.sum(data[:, i])
    beforeEs.append(avg)
    print(avg)

after = data[:, 1] * 0
afterEs = []
for i in range(6, 11):
    after += data[:, i]
    avg = np.sum(E * data[:, i]) / np.sum(data[:, i])
    afterEs.append(avg)
    print(avg)

print(scipy.stats.ttest_ind(beforeEs, afterEs))

#plt.plot(E, before)
#plt.plot(E, after)
#plt.plot(E, (before - after) / np.sqrt((before+after)/2), ".")
#plt.grid()
#plt.show()

#plt.hist((before - after) / np.sqrt((before+after)/2), bins=100)
#plt.show()

wps = np.zeros((10, 10))
tps = np.zeros((10, 10))
for i in range(1, 11):
    wps[i-1,i-1] = 1
    tps[i-1,i-1] = 1
    for j in range(i+1, 11):
        _, wp = scipy.stats.wilcoxon(data[:, i], data[:, j])
        _, tp = scipy.stats.ttest_rel(data[:, i], data[:, j])
        wps[i-1, j-1] = wp
        wps[j-1, i-1] = wp
        tps[i-1, j-1] = tp
        tps[j-1, i-1] = tp

plt.imshow(tps)
plt.colorbar()
plt.show()

plt.imshow(wps)
plt.colorbar()
plt.show()

print(scipy.stats.ttest_rel(before, after))
print(scipy.stats.wilcoxon(before, after))
