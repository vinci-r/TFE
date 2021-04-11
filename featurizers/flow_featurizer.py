import scapy
import numpy as np
from scapy.all import PcapReader

PSH_mask = 0x08

def deal_with_flow(prot, nb):
	
	with open(prot +"_flows/flow" + str(nb) + ".pcap", 'rb') as f:
		
		pkts = PcapReader(f)
			
		checksums = []
		time_deltas = []
		payload_lengths = []
		PSHs = []
		
		time_prev = 0
		time = 0
		first_iter  = True

		for pkt in pkts:
			
			time_delta = 0
			time_prev = time
			time = pkt.time

			if first_iter:
				
				portA = pkt.payload.payload.sport
				portB = pkt.payload.payload.dport
				proto = pkt.payload.proto

				[portA, portB] = sorted([portA, portB]) 
				first_iter = False

			else:
				time_delta = time - time_prev

			PSH = 0
			if(type(pkt.payload.payload) is scapy.layers.inet.TCP):
				if(pkt.payload.payload.flags & PSH_mask):
					PSH = 1

			checksums.append(pkt.payload.payload.chksum)
			time_deltas.append(time_delta)
			payload_lengths.append(len(pkt.payload.payload.payload))
			PSHs.append(PSH)

		time_deltas = list(map(float, time_deltas))

		print(str(proto) + "," + str(portA) + "," +\
		       str(portB) + "," +\
		       str(np.average(checksums)) + "," +\
		       str(np.average(time_deltas)) + "," +\
		       str(np.average(payload_lengths)) + "," +\
		       str(len(np.nonzero(PSHs)[0])/len(PSHs)))

print("# proto,port1,port2,avg_chksum,avg_time_delta,avg_payload_len,PSH_prop")

for i in range(0, 1545)	:
 	deal_with_flow("tcp", i)

for i in range(0, 1):
	deal_with_flow("udp", i)
