from os import system

for i in range(0, 2215):
	system("tshark -nr merged.pcap -2 -R " + \
                '"tcp.stream eq ' + str(i) + '" -w flows/flow' + str(i) + '.pcap')
