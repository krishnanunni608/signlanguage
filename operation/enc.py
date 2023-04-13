from cryptography.fernet import Fernet
def sec(message,keys):
    with open(keys, 'rb') as mykey:
        key = mykey.read()
        f = Fernet(key)
    encrypted = f.encrypt(message)
    return encrypted
def secd(encrypted,keys):
    with open(keys, 'rb') as mykey:
        key = mykey.read()
        f = Fernet(key)
    d = f.decrypt(encrypted)
    return  d
