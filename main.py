#from utils import generate_random_start, generate_from_seed
import os
from app import app
import urllib.request
from utils import image_process
from flask import Flask, render_template, request, redirect, flash
# from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField
from werkzeug.utils import secure_filename
from aster_pytorch.demo import create_model, predict

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'model_best.pth.tar')
# model = app.config['MODEL']
# encoder= app.config['ENCODER']
#create model
model =  create_model(resume= MODEL_DIR, encoders = 2, decoders = 2)
# Create app
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            flash('File successfully uploaded')
            img = image_process(filepath)
            kq, _ = predict(model, img)
            response = {}
            response['path'] = 'static/images/'+filename
            response['text'] = kq 
            return render_template('index.html', response=response)

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    #load_keras_model()
    # Run app
    app.run(host="0.0.0.0", port=8081, debug=False )