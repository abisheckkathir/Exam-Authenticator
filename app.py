from flask import Flask, flash, redirect, render_template, Response, request, session, abort
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import shutil
import pandas as pd
import csv
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
import pandas as pd
from time import time
import keras
from skimage import io
from skimage.filters import threshold_otsu  
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

a = 0
usern = "Unknown"

def encode():
    
    
    imagePaths = list(paths.list_images('Images'))
    knownEncodings = []
    knownNames = []
    
    for (i, imagePath) in enumerate(imagePaths):
        
        name = imagePath.split(os.path.sep)[-2]
        
        
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model='hog')
        
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    
    data = {"encodings": knownEncodings, "names": knownNames}
    
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()


def recog():
    
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    
    faceCascade = cv2.CascadeClassifier(cascPathface)
    
    data = pickle.loads(open('face_enc', "rb").read())
    a = 0
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        encodings = face_recognition.face_encodings(rgb)
        names = []
        
        
        for encoding in encodings:
            
            
            
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            
            name = "Unknown"
            
            if True in matches:
                
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                
                for i in matchedIdxs:
                    
                    name = data["names"][i]
                    
                    counts[name] = counts.get(name, 0) + 1
                
                name = max(counts, key=counts.get)

            
            names.append(name)
            
            for ((x, y, w, h), name) in zip(faces, names):
                
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            a += 1
            global usern
            usern = name
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()


genuine_image_paths = "Dataset/real/"
forged_image_paths = "Dataset/forged/"


def cap():
    while True:
        ret, frame = camera.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def regImage(un):
    if not (os.path.exists('Images/' + un)):
        os.mkdir('Images/' + un)
        
        
        
        
        
        cam=cv2.VideoCapture(0)
        for i in range(15):
            return_value, image = cam.read()
            if i > 4:
                cv2.imwrite('Images/' + un + '/' + un + str(i - 4) + '.png', image)





def rgbgrey(img):
    
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg


def greybin(img):
    
    blur_radius = 0.8
    
    img = ndimage.gaussian_filter(img, blur_radius)
    
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg


def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img)  
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey)  
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg == 1)
    
    
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    return signimg






def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                a = a + 1
    total = img.shape[0] * img.shape[1]
    return a / total


def Centroid(img):
    numOfWhites = 0
    a = np.array([0, 0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                b = np.array([row, col])
                a = np.add(a, b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites
    centroid = centroid / rowcols
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity


def SkewKurtosis(img):
    h, w = img.shape
    x = range(w)  
    y = range(h)  
    
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2
    sx = np.sqrt(np.sum(x2 * xp) / np.sum(img))
    sy = np.sqrt(np.sum(y2 * yp) / np.sum(img))

    
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3
    skewx = np.sum(xp * x3) / (np.sum(img) * sx ** 3)
    skewy = np.sum(yp * y3) / (np.sum(img) * sy ** 3)

    
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    
    kurtx = np.sum(xp * x4) / (np.sum(img) * sx ** 4) - 3
    kurty = np.sum(yp * y4) / (np.sum(img) * sy ** 4) - 3

    return (skewx, skewy), (kurtx, kurty)


def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal


def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2],
                temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features





def makeCSV():
    if not (os.path.exists('Dataset/Features')):
        os.mkdir('Dataset/Features')
        print('New folder "Features" created')
    if not (os.path.exists('Dataset/Features/Training')):
        os.mkdir('Dataset/Features/Training')
        print('New folder "Features/Training" created')
    if not (os.path.exists('Dataset/Features/Testing')):
        os.mkdir('Dataset/Features/Testing')
        print('New folder "Features/Testing" created')
    
    gpath = genuine_image_paths
    
    fpath = forged_image_paths
    n=0
    with open('users.csv', 'r+') as file:
        reader = csv.reader(file)
        n = len(list(reader))-1
        print(str(n)+'users')
        for person in range(1, n+1):
            per = ('00' + str(person))[-3:]
            print('Saving features for person id-', per)
            if not(os.path.exists('Dataset/Features/Training/training_' + per + '.csv')):
                with open('Dataset/Features/Training/training_' + per + '.csv', 'w') as handle:
                    handle.write(
                        'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
                    
                    for i in range(0, 3):
                        source = os.path.join(gpath, per + per + '_00' + str(i) + '.png')
                        features = getCSVFeatures(path=source)
                        handle.write(','.join(map(str, features)) + ',1\n')
                    for i in range(0, 3):
                        source = os.path.join(fpath, '021' + per + '_00' + str(i) + '.png')
                        features = getCSVFeatures(path=source)
                        handle.write(','.join(map(str, features)) + ',0\n')
            if not(os.path.exists('Dataset/Features/Testing/testing_' + per + '.csv')):
                with open('Dataset/Features/Testing/testing_' + per + '.csv', 'w') as handle:
                    handle.write(
                        'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
                    
                    for i in range(3, 5):
                        source = os.path.join(gpath, per + per + '_00' + str(i) + '.png')
                        features = getCSVFeatures(path=source)
                        handle.write(','.join(map(str, features)) + ',1\n')
                    for i in range(3, 5):
                        source = os.path.join(fpath, '021' + per + '_00' + str(i) + '.png')
                        features = getCSVFeatures(path=source)
                        handle.write(','.join(map(str, features)) + ',0\n')





def testing(path):
    feature = getCSVFeatures(path)
    if not (os.path.exists('Dataset/TestFeatures')):
        os.mkdir('Dataset/TestFeatures')
    with open('Dataset/TestFeatures/testcsv.csv', 'w') as handle:
        handle.write(
            'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature)) + '\n')


def readCSV(train_path, test_path, type2=False):
    
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    
    train_input = train_input.astype(np.float32, copy=False)
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(
        correct, 2)  
    
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not (type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(
            correct, 2)  
    if not (type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input



def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer


def evaluate(train_path, test_path, type2=False):
    if not (type2):
        train_input, corr_train, test_input, corr_test = readCSV(
            train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(
            train_path, test_path, type2)
    ans = 'Random'
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(training_epochs):
            
            _, cost = sess.run([train_op, loss_op], feed_dict={
                X: train_input, Y: corr_train})
            if cost < 0.0001:
                break
        
        
        
        

        
        accuracy1 = accuracy.eval({X: train_input, Y: corr_train})
        
        
        if type2 is False:
            accuracy2 = accuracy.eval({X: test_input, Y: corr_test})
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            print(prediction)
            if prediction[0][1] > prediction[0][0]:
                print('Genuine Image')
                ver = True
                return True
            else:
                print('Forged Image')
                ver = False
                return False


def trainAndTest(rate=0.001, epochs=1700, neurons=7, display=False):
    start = time()

    
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    
    n_hidden_1 = neurons  
    n_hidden_2 = 7  
    n_hidden_3 = 30  

    train_avg, test_avg = 0, 0
    n = 10
    for i in range(1, n + 1):
        if display:
            print("Running for Person id", i)
        temp = ('0' + str(i))[-2:]
        train_score, test_score = evaluate(train_path.replace(
            '01', temp), test_path.replace('01', temp))
        train_avg += train_score
        test_avg += test_score
    if display:
        
        print("Training average-", train_avg / n)
        print("Testing average-", test_avg / n)
        print("Time taken-", time() - start)
    return train_avg / n, test_avg / n, (time() - start) / n





def signrecog(uid, filename):
    global n_input
    n_input = 9
    global train_person_id
    train_person_id = uid
    global test_image_path
    test_image_path = filename
    global train_path
    train_path = 'Dataset/Features/Training/training_' + \
                 train_person_id + '.csv'
    testing(test_image_path)
    global test_path
    test_path = 'Dataset/TestFeatures/testcsv.csv'
    tf.reset_default_graph()
    
    global learning_rate
    learning_rate = 0.001
    global training_epochs
    training_epochs = 1000
    global display_step
    display_step = 1

    
    global n_hidden_1
    n_hidden_1 = 7  
    global n_hidden_2
    n_hidden_2 = 10  
    global n_hidden_3
    n_hidden_3 = 30  
    global n_classes
    n_classes = 2  

    
    global X
    X = tf.placeholder("float", [None, n_input])
    global Y
    Y = tf.placeholder("float", [None, n_classes])

    
    global weights
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
    }
    global biases
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes], seed=4))
    }
    
    global logits
    logits = multilayer_perceptron(X)

    
    global loss_op
    loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
    global optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global train_op
    train_op = optimizer.minimize(loss_op)
    
    global pred
    pred = tf.nn.softmax(logits)  
    global correct_prediction
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    global accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    global init
    init = tf.global_variables_initializer()
    return evaluate(train_path, test_path, type2=True)

def register_validation(user):
    with open('users.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[1] == user:
                return 1
    return 0

app = Flask(__name__)
app.secret_key = 'Examination Portal'





@app.route('/')
def start():
    return render_template("login.html")


#database = {'002': '123', 'abisheck': '123'}


@app.route('/login', methods=['POST', 'GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    with open('users.csv', 'r+') as file:
        reader = csv.reader(file)
        a=0
        for row in reader:
            if a == 0:
                a+=1
                continue
            if row[1]==name1:
                if row[2]==pwd:
                    global session
                    session['username'] = name1
                    session['uid']=row[0]
                    return redirect("/cam", code=302)
                else:
                    return render_template('login.html', info='Invalid username or password')
    return render_template('login.html', info='Invalid username or password')

@app.route('/cam')
def cam():
    return (render_template("cam.html", user=session['username']))


@app.route('/capture', methods=['POST', 'GET'])
def capture():
    if usern == session['username']:
        return redirect("/upload", code=302)
    print(usern)
    print(session['username'])


@app.route('/sign', methods=['POST', 'GET'])
def sign():
    if request.method == 'POST':  
        f = request.files['file']
        f.save(os.path.join('Images/temp.png'))
    print('saved')
    uid = session['uid']
    path =os.path.join('Images/temp.png')
    if signrecog(uid, path):
        os.remove(path)
        return redirect("/exam", code=302)
    else:
        return redirect("/error", code=302)


@app.route('/upload')
def upload():
    return (render_template("upload.html", user=session['username']))


@app.route('/exam')
def exam():
    return (render_template("exam.html", user=session['username']))

@app.route('/reg_image', methods=['POST', 'GET'])
def reg_image():
    camera.release()
    uname=session['username']
    regImage(uname)
    return redirect("/register3", code=302)

@app.route('/video_feed')
def video_feed():
    return Response(recog(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/register_video_feed')
def register_video_feed():
    return Response(cap(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/register1')
def register1():
    return (render_template("regi.html"))

@app.route('/register2', methods=['POST', 'GET'])
def register2():
    if(request.form['pass']==request.form['cpass']):
        global session
        session['username']=request.form['uname']
        session['password']=request.form['pass']
        flag=register_validation(session['username'])
        if flag == 1:
            return (render_template("regi.html", info='Username is not available'))
        global camera
        camera = cv2.VideoCapture(0)
        return (render_template("regi2.html"))
    else:
        return (render_template("regi.html",info='Passwords do not match'))

@app.route('/register3')
def register3():
    return (render_template("regi3.html"))


@app.route('/get_sign', methods=['POST', 'GET'])
def get_sign():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        with open('users.csv', 'r+',newline='') as file:
            reader = csv.reader(file)
            n = str(len(list(reader))).zfill(3)
            session['uid']=n
            writer = csv.writer(file,delimiter=',',lineterminator='\n')
            writer.writerow([n, session['username'],session['password']])
        print(len(files))
        uid=session['uid']
        for i in range(0,5):
            source = os.path.join(genuine_image_paths, uid + uid + '_00' + str(i) + '.png')
            files[i].save(source)
    print('saved')
    makeCSV()
    encode()
    return redirect("/success", code=302)

'''
@app.route('/get_sign', methods=['POST', 'GET'])
def get_sign():
    if request.method == 'POST':
        if request.form['submit']=='Submit':
            files = request.files.getlist('files[]')
            print(len(files))
            uid=session['uid']
            for i in range(0,5):
                source = os.path.join(genuine_image_paths, uid + uid + '_00' + str(i) + '.png')
                files[i].save(source)
            print('saved')
            makeCSV()
            encode()
            return redirect("/success", code=302)
        elif request.form['submit'] == 'Reset':
            f = open("users.csv", "r+")
            lines = f.readlines()
            lines = lines[:-1]
            writer = csv.writer(f, delimiter=',')
            for line in lines:
                writer.writerow(line)
            user=session['username']
            directory = user
            parent= "Images/"
            path=os.path.join(parent, directory)
            shutil.rmtree(path, ignore_errors=True)
            return redirect("/register1", code=302)
'''

@app.route('/success')
def success():
    return (render_template("success.html"))

@app.route('/logout')
def logout():
    global usern
    usern = "Unknown"
    session.pop('username', None)
    session.pop('uid', None)
    session.pop('password', None)
    return (render_template("login.html"))

@app.route('/error')
def error():
    return (render_template("error.html"))

@app.errorhandler(500)
def internal_error(error):
    return redirect("/error", code=302)

if __name__ == '_main_':
    app.run(debug=True)