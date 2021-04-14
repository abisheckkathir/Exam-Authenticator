from flask import Flask, flash, redirect, render_template, request, session, abort


app = Flask(__name__)
app.secret_key = 'Examination Portal'

@app.route('/')
def start():
    return render_template("login.html")
database={'STU001':'123', 'STU002':'457'}

@app.route('/login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    session['username'] = name1
    if name1 not in database:
	    return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
	         return render_template('cam.html',name=name1,user=session['username'])


@app.route('/cam')
def cam():
    return(render_template("cam.html",user=session['username']))


@app.route('/upload')
def upload():
    return(render_template("upload.html",user=session['username']))

@app.route('/exam')
def exam():
    return(render_template("exam.html",user=session['username']))


@app.route('/logout')
def logout():
    session.pop('username', None)
    return(render_template("login.html"))

if __name__ == '_main_':
    app.run()
