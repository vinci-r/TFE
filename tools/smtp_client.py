import smtplib
import random

senders_pool = ["Mark Smith", "Bill Overbeck", "Gini Weasley",\
                "Henry VIII", "Ludwig Van Beethoven"]

senders_addresses = ["mark.smith@yahoo.fr", "overbill@gmail.com", 
                      "gini@hogwarts.co.uk", "henry@king.com",
                       "ludwig@spotify.com"]

receivers_pool = ["Robert House", "Bill Gates", "Gerard Depardieu",\
                  "Mickey Mouse", "Edward Cullen"]

receivers_addresses = ["bobby@house.us", "bill.gates@microsoft.com", 
                        "gerard.depardieu@telefrance.fr", "mickey@disney.com",
                        "edward.cullen@twillight.tv"]

content_headers = ["Dear Sir or Madam,\n\n", "Mom,\n\n",
                  "Dear Customer,\n\n", "Your Majesty,\n\n",
                  "Esteemed friend,\n\n"]

content_core = ["Please call me as soon a possible, this is urgent.\n\n",
                "Come home, you forgot your lunch and your ACDC concert tickets have arrived.\n\n",
                "I am pleased to invite you to my Vegas venue for your recital, search for the Cesar Palace on Google Maps, you'll find it.\n\n",
                "The sales numbers are way too low, if this continues, I'll have no choice but to part ways with you\n\n",
                "Bella went on a crazy roadtrip with Jacob, please stop her before this turns into a Thelma and Louise situation.\n\n"]

signature = ["Yours truly,\n","With pleasure,\n", "Until tomorrow,\n",
             "Do you job,\n", "With love,\n"]


for i in range(200):
	smtpObj = smtplib.SMTP('192.168.0.3')

	rand = random.randint(0,4)
	sender = senders_addresses[rand]
	sender_name = senders_pool[rand]

	rand = random.randint(0,4)
	receivers = [receivers_addresses[rand]]
	receiver_name = receivers_pool[rand]

	message_content = "" + content_headers[random.randint(0,4)]
	message_content += content_core[random.randint(0,4)]
	message_content += signature[random.randint(0,4)]
	message_content += sender_name

	message = "From: " + sender_name + " <" + sender + "> "
	message += "To: " + receiver_name + " <" + receivers[0] + "> \n\n"
	message += message_content
	
	print(message)

	smtpObj.sendmail(sender,receivers, message)
	print("Email sent successfully")


