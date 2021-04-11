from ftplib import FTP

ftp = FTP('192.168.0.3')
ftp.login()

fileList =  open("list.txt", 'r')
lines = fileList.readlines()

for i in lines:
	with open(i[0:-1], 'wb') as fp:
		ftp.retrbinary("RETR " + i[0:-1], fp.write)

ftp.quit()