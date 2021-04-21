from flask import Flask, flash, redirect, render_template, Response, request, session, abort
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import csv
from tkinter import *
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
from skimage.filters import threshold_otsu   # For finding the threshold for grayscale to binary conversion
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


a = 0


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
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
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
        # Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb, model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    # use pickle to save data into a file for later use
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()


def recog():
    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    a = 0
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
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            # set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
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
            a += 1
            global usern
            usern = name
            # return name
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
genuine_image_paths = "Dataset/real/"
forged_image_paths = "Dataset/forged/"


# ## Preprocessing the image


def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg


def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    # to remove small components or noise
    img = ndimage.gaussian_filter(img, blur_radius)
#     img = ndimage.binary_erosion(img).astype(img.dtype)
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
    grey = rgbgrey(img)  # rgb to grey
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey)  # grey to binary
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg == 1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    return signimg


# ## Feature Extraction
#


def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                a = a+1
    total = img.shape[0] * img.shape[1]
    return a/total


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
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity


def SkewKurtosis(img):
    h, w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    # calculate projections along the x and y axes
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    # centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    # standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))

    # skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    # Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

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


# ## Saving the features


def makeCSV():
    if not(os.path.exists('Dataset/Features')):
        os.mkdir('Dataset/Features')
        print('New folder "Features" created')
    if not(os.path.exists('Dataset/Features/Training')):
        os.mkdir('Dataset/Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('Dataset/Features/Testing')):
        os.mkdir('Dataset/Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    for person in range(1, 13):
        per = ('00'+str(person))[-3:]
        print('Saving features for person id-', per)

        with open('Dataset/Features/Training/training_'+per+'.csv', 'w') as handle:
            handle.write(
                'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0, 3):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0, 3):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')

        with open('Dataset/Features/Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write(
                'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(3, 5):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(3, 5):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')





# # TF Model


def testing(path):
    feature = getCSVFeatures(path)
    if not(os.path.exists('Dataset/TestFeatures')):
        os.mkdir('Dataset/TestFeatures')
    with open('Dataset/TestFeatures/testcsv.csv', 'w') as handle:
        handle.write(
            'ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')





def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    # Converting input to float_32
    train_input = train_input.astype(np.float32, copy=False)
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(
        correct, 2)      # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(
            correct, 2)      # Converting to one hot
    if not(type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input





# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer




def evaluate(train_path, test_path, type2=False):
    if not(type2):
        train_input, corr_train, test_input, corr_test = readCSV(
            train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(
            train_path, test_path, type2)
    ans = 'Random'
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={
                               X: train_input, Y: corr_train})
            if cost < 0.0001:
                break
#             # Display logs per epoch step
#             if epoch % 999 == 0:
#                 print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
#         print("Optimization Finished!")

        # Finding accuracies
        accuracy1 = accuracy.eval({X: train_input, Y: corr_train})
#         print("Accuracy for train:", accuracy1)
#         print("Accuracy for test:", accuracy2)
        if type2 is False:
            accuracy2 = accuracy.eval({X: test_input, Y: corr_test})
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            print(prediction)
            if prediction[0][1] > prediction[0][0]:
                print('Genuine Image')
                ver=True
                return True
            else:
                print('Forged Image')
                ver=False
                return False


def trainAndTest(rate=0.001, epochs=1700, neurons=7, display=False):
    start = time()

    # Parameters
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    # Network Parameters
    n_hidden_1 = neurons  # 1st layer number of neurons
    n_hidden_2 = 7  # 2nd layer number of neurons
    n_hidden_3 = 30  # 3rd layer

    train_avg, test_avg = 0, 0
    n = 10
    for i in range(1, n+1):
        if display:
            print("Running for Person id", i)
        temp = ('0'+str(i))[-2:]
        train_score, test_score = evaluate(train_path.replace(
            '01', temp), test_path.replace('01', temp))
        train_avg += train_score
        test_avg += test_score
    if display:
        #         print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Training average-", train_avg/n)
        print("Testing average-", test_avg/n)
        print("Time taken-", time()-start)
    return train_avg/n, test_avg/n, (time()-start)/n
makeCSV()

def signrecog(uid,filename):
    global n_input
    n_input = 9
    global train_person_id
    train_person_id = uid
    global test_image_path
    test_image_path = filename
    global train_path
    train_path = 'Dataset/Features/Training/training_' + \
        train_person_id+'.csv'
    testing(test_image_path)
    global test_path
    test_path = 'Dataset/TestFeatures/testcsv.csv'
    tf.reset_default_graph()
    # Parameters
    global learning_rate
    learning_rate = 0.001
    global training_epochs
    training_epochs = 1000
    global display_step
    display_step = 1

    # Network Parameters
    global n_hidden_1
    n_hidden_1 = 7  # 1st layer number of neurons
    global n_hidden_2
    n_hidden_2 = 10  # 2nd layer number of neurons
    global n_hidden_3
    n_hidden_3 = 30  # 3rd layer
    global n_classes
    n_classes = 2  # no. of classes (genuine or forged)

    # tf Graph input
    global X
    X = tf.placeholder("float", [None, n_input])
    global Y
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
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
    # Construct model
    global logits
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    global loss_op
    loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
    global optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global train_op
    train_op = optimizer.minimize(loss_op)
    # For accuracies
    global pred
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    global correct_prediction
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    global accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Initializing the variables
    global init
    init = tf.global_variables_initializer()
    return evaluate(train_path, test_path, type2=True)

encode()
app = Flask(__name__)
app.secret_key = 'Examination Portal'


# @app.route('/')
# def index():
#     return render_template('cam.html')
@app.route('/')
def start():
    return render_template("login.html")


database = {'Siva': '123', 'abisheck': '123'}


@app.route('/login', methods=['POST', 'GET'])
def login():
    name1 = request.form['username']
    pwd = request.form['password']
    global session
    session['username'] = name1
    if name1 not in database:
        return render_template('login.html', info='Invalid User')
    else:
        if database[name1] != pwd:
            return render_template('login.html', info='Invalid Password')
        else:
            # return render_template('cam.html',name=name1,user=session['username'])
            return redirect("/cam", code=302)


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
    filename = request.form['file']
    uid = request.form['uid']
    path="C:/Users/Sivasini/Downloads/"+filename
    if signrecog(uid,path):
        return redirect("/exam", code=302)
    else :
        return redirect("/error", code=302)

@app.route('/upload')
def upload():
    return (render_template("upload.html", user=session['username']))


@app.route('/exam')
def exam():
    return (render_template("exam.html", user=session['username']))


@app.route('/video_feed')
def video_feed():
    return Response(recog(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return (render_template("login.html"))

@app.route('/error')
def error():
    return (render_template("error.html"))

@app.errorhandler(500)
def internal_error(error):
    return redirect("/error", code=302)

if __name__ == '_main_':
    app.run()