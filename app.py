from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps
from PIL.Image import Resampling
from flask import Flask, redirect, url_for, request, render_template , session 
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_mysqldb import MySQL
import shutil
import MySQLdb.cursors

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'nail'
mysql = MySQL(app)

MODEL_PATH = 'models/Keras_model.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()          
print('Model loaded. Check http://127.0.0.1:5000/')

class_names = ["Acral Lentiginous Melanoma", 
               "Beaus Line", 
               "Blue Finger", 
               "Clubbing", 
               "Healthy Nail", 
               "Koilonychia", 
               "Muehrckes Lines",  
               "Pitting", 
               "Terrys Nail",
               "Error-Not Nail"]
               
def model_predict(img_path, model):

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(img_path).convert("RGB")

    size = (224, 224)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    predictions = model.predict(data)[0]

    max_index = predictions.argmax()

    max_class_name = class_names[max_index].strip()

    max_probability = predictions[max_index]

    max_percentage = max_probability * 100

    output_list = []
    for i in range(len(class_names)):
        class_name = class_names[i].strip()
        probability = predictions[i]
        percentage = probability * 100
        output_list.append(f"{class_name}: {percentage:.2f} %")
    output_list.append(max_class_name)

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users_data ORDER BY id DESC LIMIT 1")
    latest_row = cur.fetchone()

    if latest_row:
        latest_id = latest_row[0]
        new_name = max_class_name  
        cur.execute("UPDATE users_data SET name = %s WHERE id = %s", (max_class_name, latest_id))
        mysql.connection.commit()
        cur.close()
    
    return output_list , str(max_class_name)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/Diseases')
def Diseases():

    return render_template('Diseases.html')

@app.route('/Contect')
def Contect():

    return render_template('Contect.html')
    
@app.route('/Contect' , methods=['POST'])
def submit1():
    ids = request.form['ids']
    name = request.form['name']
    comments = request.form['comments']

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO comments (ids , name , comments) VALUES (%s,%s,%s)", (ids , name , comments))
    mysql.connection.commit()
    cur.close()
    return render_template('index.html')

@app.route('/Form')
def Form():

    return render_template('Form.html')

@app.route('/Form', methods=['POST'])
def submit():
    # Get form data
    FirstName = request.form['FirstName']
    SecondName = request.form['SecondName']
    Age = request.form['Age']
    Gender = request.form['Gender']
    Email = request.form['Email']
    kidlivprob = request.form['kidlivprob']
    presure = request.form['presure']
    diabetes  = request.form['diabetes']
    anemia = request.form['anemia']
    heart = request.form['heart']

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO users_data (FirstName, SecondName, Age, Gender, Email , kidlivprob , presure , diabetes , anemia , heart) VALUES (%s,%s,%s,%s,%s,%s, %s, %s, %s, %s)", (FirstName, SecondName, Age, Gender, Email , kidlivprob , presure , diabetes , anemia , heart))
    mysql.connection.commit()
    cur.close()
    return render_template('view.html')

@app.route('/Notes')
def Notes():
    return render_template('Notes.html')
    
@app.route('/predict')
def predict():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users_data ORDER BY id DESC LIMIT 1")
    data = cur.fetchall()
    cur.close()
    return render_template('predict.html', data=data)

@app.route('/Search')
def Search():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users_data")
    data = cur.fetchall()
    cur.close()
    return render_template('Search.html', data=data)

@app.route('/search2', methods=['GET', 'POST'])
def search2():
    id = request.form['id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users_data WHERE id = %s", (id,))
    user = cur.fetchall()
    cur.close()
    return render_template('search2.html', data=user)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        filename = secure_filename(f.filename)
        # Add the string 'img#$21' to the filename before the extension
        new_filename = os.path.splitext(filename)[0] + 'img#$21' + os.path.splitext(filename)[1]
        file_path = os.path.join(basepath, 'uploads', new_filename)        
        f.save(file_path)

        preds, maxs = model_predict(f, model)
        if maxs == "Healthy Nail":
            dest_dir = os.path.join(basepath, 'Healthy Nail')
        elif maxs == "Acral Lentiginous Melanoma":
            dest_dir = os.path.join(basepath, 'Acral Lentiginous Melanoma')
        elif maxs == "Beaus Line":
            dest_dir = os.path.join(basepath, 'Beaus Line')
        elif maxs == "Blue Finger":
            dest_dir = os.path.join(basepath, 'Blue Finger')
        elif maxs == "Clubbing":
            dest_dir = os.path.join(basepath, 'Clubbing')
        elif maxs == "Koilonychia":
            dest_dir = os.path.join(basepath, 'Koilonychia')
        elif maxs == "Muehrckes Lines":
            dest_dir = os.path.join(basepath, 'Muehrckes Lines')
        elif maxs == "Pitting":
            dest_dir = os.path.join(basepath, 'Pitting')
        elif maxs == "Terrys Nail":
            dest_dir = os.path.join(basepath, 'Terrys Nail')
        elif maxs == "Error-Not Nail":
            dest_dir = os.path.join(basepath, 'Not Nail')

        # Create the destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Build the destination file path by joining the destination directory and the new file name
        dest_file_path = os.path.join(dest_dir, new_filename)

        # Copy the image file with the new file name from the original path to the destination path
        shutil.copyfile(file_path, dest_file_path)

        return preds
        return render_template('predict.html', maxs=maxs)
    return None



if __name__ == '__main__':
    app.run(debug=True)

