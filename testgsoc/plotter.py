import matplotlib.pyplot as plt

sizes = []
gpupartime = []
naivesingletime = []
naivemultitime = []
cputime = []

with open("data.txt") as f:
    for line in f:
        a, b, c, d, e = [float(x) for x in line.split()]
        sizes.append(a)
        gpupartime.append(b)
        naivesingletime.append(c)
        naivemultitime.append(d)
        cputime.append(e)


#plt.plot(gpupartime, sizes, label="Parallelly on GPU")
plt.plot(naivesingletime, sizes, label="Sequentially on single GPU thread")
plt.plot(naivemultitime, sizes, label="Sequentially on multiple GPU threads")
plt.plot(cputime, sizes, label="Sequentially on CPU")
plt.xlabel("Time (nanoseconds)")
plt.ylabel("Array size")

plt.legend()
plt.show()
