from flask import Flask, render_template,request,redirect,url_for,Response
from operation import fire,male,enc,cnnrecog
from collections import  namedtuple
import cv2
import imageio
global glorev
glorev='0'
user=namedtuple('user',['slno','email','firstname','lastname','password'])
app=Flask(__name__)
flag1=0
@app.route('/')
def intex():
    return render_template('test.html')
@app.route('/login', methods=["POST","GET"])
def loginpage():
    if request.method == "POST":
        username0=request.form.get('username1')
        password0=request.form.get('password1')
        passs=fire.getem(username0)
        if(passs == password0):
            return redirect(url_for('stop'))
        else:
            return redirect(url_for('loginpage'))
    else:
        return render_template('login.html')
@app.route('/signup', methods=["POST","GET"])
def signup():
    if request.method == "POST" :
        username=request.form.get('username2')
        firstname=request.form.get('firstname2')
        lastname=request.form.get('lastname2')
        password=request.form.get('password2')
        user1= user(1,username,firstname,lastname,password)
        fire.add(user1)
        return redirect(url_for('loginpage'))
    else:
        return render_template("signup.html")
@app.route('/forgot', methods=["POST","GET"])
def forgot():
    if request.method=="POST":
        username5=request.form.get('username3')
        if username5 =='':
            return  render_template("forgot.html")
        rr = bytes(username5, 'utf-8')
        encrypted=enc.sec(rr,'key/usskey.key')
        output = encrypted.decode()
        male.emsent(output)
        return redirect(url_for('loginpage'))
    else:
       return render_template("forgot.html")
@app.route('/main/<reset1>',methods=["POST","GET"])
def paass(reset1):
    if request.method=="POST":
        rr = bytes(reset1, 'utf-8')
        d=enc.secd(rr,'key/usskey.key')
        output = d.decode()
        username51=request.form.get('password12')
        fire.updatepassword(output,username51)
        return redirect(url_for('loginpage'))
    else:
        return render_template("passwordreset.html")

#index page
@app.route('/stop')
def stop():
    return render_template('testi.html')


@app.route('/main')
def start():
    return render_template('index.html')
#cnn recognition integration(SLR SYSTEM)
@app.route('/cnnrec')
def index():
    return render_template('main.html')


@app.route('/video_feed')
def video_feed():
    return Response(cnnrecog.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')






def gen(l):
        path=('Reverse sign images//')
        img=imageio.imread(path+str(l)+'.jpg')
        ret, buffer = cv2.imencode('.jpg',img )
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/reverse',methods=["GET","POST"])
def reverse():
    if request.method=="POST":
        global glorev
        print(glorev)
        glorev= str(request.form.get('rev'))
        return render_template('reverse.html')
    else:
        return render_template('reverse.html')
@app.route('/rev_vid')
def rev_video():
    return Response(gen(glorev), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run()