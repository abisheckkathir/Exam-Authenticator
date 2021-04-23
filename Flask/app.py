from flask import Flask, flash, redirect, render_template,Response, request, session, abort
import cv2
from imutils import paths
import face_recognition
import pickle
import os
import csv
import numpy as np
import os
a=0
# camera = cv2.VideoCapture(0)

# def gen_frames():  
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def encode():
    #get paths of each file in folder named Images
    #Images here contains my data(folders of various persons)
    imagePaths = list(paths.list_images('Images'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()
def recog():
    
    #find path of xml file containing haarcascade file 
    cascPathface = os.path.dirname(
     cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    a=0
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
           #Compare encodings with encodings in data["encodings"]
           #Matches contain array with boolean values and True for the embeddings it matches closely
           #and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
             encoding)
            #set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)


            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                 0.75, (0, 255, 0), 2)
            a+=1
            global usern
            usern=name
        if a==5:
            video_capture.release()      
            # return name
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
encode()
app = Flask(__name__)
app.secret_key = 'Examination Portal'
# @app.route('/')
# def index():
#     return render_template('cam.html')
@app.route('/')
def start():
    return render_template("login.html")
database={'abisheck':'123', 'STU002':'457'}

@app.route('/login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    global session
    session['username'] = name1
    if name1 not in database:
	    return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
            # return render_template('cam.html',name=name1,user=session['username'])
            return redirect("/cam", code=302)


@app.route('/cam')
def cam():
    return(render_template("cam.html",user=session['username']))

@app.route('/capture',methods=['POST','GET'])
def capture():
    if usern==session['username']:
        return redirect("/upload", code=302)
    print(usern)
    print(session['username'])

@app.route('/upload')
def upload():
    return(render_template("upload.html",user=session['username']))

@app.route('/exam')
def exam():
    return(render_template("exam.html",user=session['username']))
@app.route('/video_feed')
def video_feed():
    return Response(recog(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return(render_template("login.html"))

if __name__ == '_main_':
    app.run()
