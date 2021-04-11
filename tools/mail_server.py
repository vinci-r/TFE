import smtpd
import asyncore

class CustomSMTPServer(smtpd.SMTPServer):
	def process_message(self, peer, mailfrom, rcpttos, data):
		print(peer, mailfrom, rcpttos, len(data))

server = CustomSMTPServer(('192.168.0.3', 25), None)
asyncore.loop()
