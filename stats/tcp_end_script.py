from scapy.all import PcapReader



print("flow,FIN_count,RST_count")
for i in range(0, 1545)	:

	with open("tcp_flows/flow" + str(i) + ".pcap", 'rb') as f:
		
		pkts = PcapReader(f)
		FIN_count = 0
		RST_count = 0
		FIN = 0x01
		RST = 0x04


		for pkt in pkts:
			flags = pkt.payload.payload.flags

			if flags & FIN:
				FIN_count += 1

			if flags & RST:
				RST_count += 1


	print(str(i) + "," + str(FIN_count) + "," + str(RST_count))