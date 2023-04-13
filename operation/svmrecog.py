def gen_frames():
    # importing necessary libraries
    import cv2
    import imutils
    import numpy as np
    import os
    from os import path
    import pickle
    import imageio
    from scipy import ndimage
    from scipy.spatial import distance
    import tensorflow as tf
    from tensorflow import keras
    import keras

    camera=cv2.VideoCapture(2)
    #global variables
    bg=None
    visual_dict={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',
             28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}
    aWeight=0.5
    #t,r,b,l=100,350,228,478

    # Global Variables
    t,r,b,l=100,350,325,575
    num_frames=0
    shape=180


    model='files/SVM'

    infile = open(model,'rb')
    svm = pickle.load(infile)
    infile.close()
    model1='files/kmeans'

    infile = open(model1,'rb')
    kmeans = pickle.load(infile)
    infile.close()

    bg=None

    #To find the running average over the background
    def run_avg(image,aweight):
        nonlocal bg #initialize the background
        if bg is None:
            bg=image.copy().astype("float")
            return
        cv2.accumulateWeighted(image,bg,aweight)

    #Segment the egion of hand
    def extract_hand(image,threshold=25):
        nonlocal bg
        diff=cv2.absdiff(bg.astype("uint8"),image)
        thresh=cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
        (_,cnts,_)=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if(len(cnts)==0):
            return
        else:
            max_cont=max(cnts,key=cv2.contourArea)
            return (thresh,max_cont)



    while (camera.isOpened()):
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if frame is not None:
                frame=imutils.resize(frame,width=700)
                frame=cv2.flip(frame,1)
                clone=frame.copy()

                roi=frame[t:b,r:l]
                gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                gray=cv2.GaussianBlur(gray,(7,7),0)

                if(num_frames<30):
                    run_avg(gray,aWeight)
                    cv2.putText(clone, "Keep the Camera still.", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                else:
                    cv2.putText(clone, "Keep the Camera still.", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                    cv2.putText(clone, "Put your hand in the rectangle", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                    cv2.putText(clone, "Press the key of the sample", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                    hand=extract_hand(gray)
                    if hand is not None:
                        thresh,max_cont=hand
                        mask=cv2.drawContours(clone,[max_cont+(r,t)],-1, (0, 0, 255))
                        mask=np.zeros(thresh.shape,dtype="uint8")
                        cv2.drawContours(mask,[max_cont],-1,255,-1)
                        mask = cv2.medianBlur(mask, 5)
                        mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        res=cv2.bitwise_and(roi,roi,mask=mask)
                        res=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

                        #---- Apply automatic Canny edge detection using the computed median----

                        high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        lowThresh = 0.5 * high_thresh
                        #res=cv2.Canny(res,lowThresh,high_thresh)

                        #cv2.imshow("Segmented",res)
                        hand=cv2.bitwise_and(gray,gray,mask=thresh)
                        #cv2.imshow("Hand",hand)
                        res = cv2.Canny(hand, lowThresh, high_thresh)

                        # CNN Model
                        surf = cv2.xfeatures2d.SURF_create()
                        kp, desc = surf.detectAndCompute(res, None)
                        #print("Surf features extracted!")
                        features = cv2.drawKeypoints(res, kp, None)
                        cv2.imshow("Surf Features",features)

                        if desc is not None:
                            visual_words=kmeans.predict(desc)
                            hist = np.array(np.bincount(visual_words,minlength=shape))
                            hist=hist.reshape(1,-1)
                            sign = svm.predict(hist)
                            output=visual_dict[sign[0]]
                            cv2.putText(clone, 'Sign ' + str(output), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 2,
                                        (0, 0, 255))

            cv2.rectangle(clone, (l, t), (r, b), (0, 255, 0), 2)
            num_frames += 1

            ret, buffer = cv2.imencode('.jpg', clone)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')