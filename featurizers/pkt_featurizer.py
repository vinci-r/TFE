import scapy
from scapy.all import PcapReader

def ip_to_int(ip):
	
	bytes = str.split(ip, ".")
	
	shifter = 1
	number = 0

	reversed_bytes = reversed(bytes)

	for i in reversed_bytes:
		number += shifter * int(i)
		shifter *= 256

	return number

PSH_mask = 0x08

with open("legit_merged2.pcap", 'rb') as f:

	pkts = PcapReader(f)
	print("# proto,dst,dport,sport,chksum,tcp_psh,time_delta,payload_len")

	time_prev = 0
	time = 0
	first_iter  = True

	for pkt in pkts:
		
		time_delta = 0
		time_prev = time
		time = pkt.time

		if(not first_iter):
			time_delta = time - time_prev


		if(type(pkt.payload) is scapy.layers.inet.IP):
			if(type(pkt.payload.payload) is scapy.layers.inet.TCP\
			      or type(pkt.payload.payload) is scapy.layers.inet.UDP):
				
				PSH = 0
				if(type(pkt.payload.payload) is scapy.layers.inet.TCP):
					if(pkt.payload.payload.flags & PSH_mask):
						PSH = 1

				print(str(pkt.payload.proto) + "," +\
					   str(ip_to_int(str(pkt.payload.dst)))+ "," +\
					   str(pkt.payload.payload.dport) + "," +\
					   str(pkt.payload.payload.sport) + "," +\
					   str(pkt.payload.payload.chksum) + "," +\
					   str(PSH) + "," +\
					   str(time_delta) + "," +\
				       str(len(pkt.payload.payload.payload)))

		first_iter = False


