def ip_to_int(ip):
	
	bytes = str.split(ip, ".")
	
	shifter = 1
	number = 0

	reversed_bytes = reversed(bytes)

	for i in reversed_bytes:
		number += shifter * int(i)
		shifter *= 256

	return number

def int_to_ip(nb):

	ip = str(nb % 256)
	nb //= 256

	while nb > 0:
		ip = str(nb % 256) + "." + ip
		nb //= 256

	return ip

print(hex(ip_to_int("192.168.1.1")))
print(int_to_ip(ip_to_int("192.168.1.1")))