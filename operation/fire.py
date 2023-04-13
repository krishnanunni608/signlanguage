import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

#initialisation of fibase account or database
cred = credentials.Certificate('C:/Users/krish/OneDrive/Desktop/final-af6e4-firebase-adminsdk-v2jpk-ff0d285191.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

#function to add user data
def add(user1):
    db.collection('person').document(user1.email).set({'slno': user1.slno,'email':user1.email,'firstname':user1.firstname,'lastname':user1.lastname,'password':user1.password})

#function to get password by email
def getem(email1):
    dict={'password': "default"}
    docs=db.collection('person').where("email","==",email1).get()
    for doc in docs:
        dict=doc.to_dict()
    return(dict['password'])

    
def updatepassword(email2,password1):
    docs=db.collection('person').where("email","==",email2).get()
    field_updates = {"password": password1}
    for item in docs:
        doc = db.collection('person').document(item.id)
        doc.update(field_updates)


