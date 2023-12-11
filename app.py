import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Model yang akan digunakan 
model_used = './model/model_rps.h5'

# *****
def predict_result(model, run_time, probs, img):
    class_list = {'Paper': 0, 'Rock': 1, 'Scissors' : 2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result.html', labels=labels,
                            probs=probs, model=model, pred=idx_pred,
                            run_time=run_time, img=img)
# ***

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/index.html', )

@app.route('/result', methods=['POST'])
def predict():
    #-----------------
    model = load_model(model_used)
    #-----------------

    file = request.files["file"]  #Mengambil file dari form di html 
    file.save(os.path.join('static', 'temp.jpg'))  # Save jadi temporay file 
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)  # Convert warna dari BGR ke RGB 
    img = np.expand_dims(cv2.resize(img, (224, 224)).astype('float32') / 255, axis=0) # Rescale 1-255 menjadi 0-1
    start = time.time()  # Mengambil waktu sekarang 
    pred = model.predict(img)[0]  # Predict Class 
    labels = (pred > 0.5).astype(int)  # Merubah dari kategorikal menjadi integer 
    runtimes = round(time.time()-start,4)  # Menghitung total waktu yang diperlukan model untuk memprediksi label 
    respon_model = [round(elem * 100, 2) for elem in pred]  
    return predict_result("MODEL KU", runtimes, respon_model, 'temp.jpg')  # Return fungsi predict result 

if __name__ == "_main_":
        app.run(debug=True, host='0.0.0.0', port=2000)