from os import system

for i in range(0, 1252):
	system("tshark -nr merged_bad2.pcap -2 -R " + \
                '"udp.stream eq ' + str(i) + '" -w udp_flows/flow' + str(i) + '.pcap')
