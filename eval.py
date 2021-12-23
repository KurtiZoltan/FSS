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

after = data[:, 1] * 0
afterEs = []
for i in range(6, 11):
    after += data[:, i]
    avg = np.sum(E * data[:, i]) / np.sum(data[:, i])
    afterEs.append(avg)

res = np.sum(afterEs) / 5
sigma = np.sqrt(1 / 4 * np.sum((afterEs-res)**2))
print(res)
print(sigma)
print((156/3 - res)/sigma)

print(scipy.stats.ttest_1samp(afterEs, 156/3))

plt.plot(E, before, label="Kalibrálás előtt")
plt.plot(E, after, label="Kalibbrálás után")
plt.xlabel("$E$ [$keV$]")
plt.ylabel("N")
plt.grid()
plt.legend()
plt.savefig("spectra.pdf")
plt.show()

#plt.hist((before - after) / np.sqrt((before+after)/2), bins=100)
#plt.show()

wps = np.zeros((10, 10))
tps = np.zeros((10, 10))
#for i in range(1, 6):
#    data[:, i] += 3
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
plt.savefig("t.pdf")
plt.show()

plt.imshow(wps)
plt.colorbar()
plt.savefig("wilcoxon.pdf")
plt.show()

print(scipy.stats.ttest_rel(before, after))
print(scipy.stats.wilcoxon(before, after))

wps = np.zeros((10, 10))
tps = np.zeros((10, 10))
for i in range(1, 6):
    data[:, i] += 3
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

plt.imshow(wps)
plt.colorbar()
plt.savefig("demo.pdf")
plt.show()
