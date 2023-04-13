from flask import Flask
from flask_mail import Mail, Message
from operation import enc
from email.mime.text import MIMEText
app = Flask(__name__)
mail= Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'samba.s.konarak@gmail.com'
app.config['MAIL_PASSWORD'] = 'hhjwmgpsojruugxh'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

def emsent(email1):
    email3=bytes(email1, 'utf-8')
    #print(email3)
    email2=enc.secd(email3,'key/usskey.key')
    email4=email2.decode()
    with app.app_context():
        msg = Message("forgot password", sender = 'samba.s.konarak@gmail.com', recipients = [email4])
        msg.body = 'FOLLOW THIS LINK TO RESET PASSWORD:'+'http://127.0.0.1:5000/main/'+email1
        mail.send(msg)