import matplotlib.pyplot as plt

sizes = []
thrusttime = []
hillistime = []

with open("data.txt") as f:
    for line in f:
        a, b, c = [float(x) for x in line.split()]
        sizes.append(a)
        thrusttime.append(b)
        hillistime.append(c)

thrusttime2 = []
hillistime2 = []
for x in range(0, 391, 10):
    for i in range(10):
        thrusttime2.append(min(thrusttime[x],thrusttime[x+1],thrusttime[x+2],thrusttime[x+3],thrusttime[x+4],thrusttime[x+5],thrusttime[x+6],thrusttime[x+7],thrusttime[x+8],thrusttime[x+9]))
        hillistime2.append(min(hillistime[x],hillistime[x+1],hillistime[x+2],hillistime[x+3],thrusttime[x+4],thrusttime[x+5],thrusttime[x+6],thrusttime[x+7],thrusttime[x+8],thrusttime[x+9]))

plt.plot(sizes, thrusttime2, label="Thrust")
#plt.plot(sizes, naivesingletime, label="Sequentially on single GPU thread")
#plt.plot(sizes, naivemultitime, label="Sequentially on multiple GPU threads")
plt.plot(sizes, hillistime2, label="Hillis-Steele")
plt.ylabel("Time (nanoseconds)")
plt.xlabel("Array size")

plt.legend()
plt.show()
