import numpy as np
from scipy import stats


nb_pkts = []
duration = []
nb_bytes = []

with open("output_tcp.conv") as f:

	for line in f:
		data = str.split(line)

		if data[0].find("192") >=0:

			nb_pkts.append(int(data[7]))
			nb_bytes.append(int(data[8]))
			duration.append(float(data[10]))

with open("output_udp.conv") as f:

	for line in f:
		data = str.split(line)

		if data[0].find("192") >=0:

			nb_pkts.append(int(data[7]))
			nb_bytes.append(int(data[8]))
			duration.append(float(data[10]))


print("NB FLOWS " + str(len(nb_pkts)))

print("PACKETS")
print(max(nb_pkts))
print(min(nb_pkts))
print(np.average(nb_pkts))
print(stats.mode(nb_pkts))

print("DURATION")
print(max(duration))
print(min(duration))
print(np.average(duration))

print("BYTES")
print(max(nb_bytes))
print(min(nb_bytes))
print(np.average(nb_bytes))
print(stats.mode(nb_bytes))


	
		


