import smtplib, ssl

from grpc import server

server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login('samaba.s.konarak@gmail.com','Krishnan@9745')
server.sendmail('sambaskonarak@gmail.com','krishnanunnijpjaya@gmail.com','testing......')
