from flask import Flask, request, jsonify, render_template, redirect
import os
import librosa
import numpy as np
from scipy.io import wavfile
from playsound import playsound
from tensorflow.keras.models import  model_from_json
import noisereduce as nr
from pydub import AudioSegment as am

UPLOAD_FOLDER = './static/'
app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

f = []
json_file = open('./static/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./static/model (1).h5")
print("Loaded model from disk")


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate= librosa.load(os.path.join(file_name),sr=48000)
    if chroma:
      stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
      mfccs=np.mean(librosa.feature.mfcc (y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result=np.hstack((result, mfccs))
    if chroma:
      chroma=np.mean(librosa.feature.chroma_stft (S=stft, sr=sample_rate).T,axis=0)
      result=np.hstack((result, chroma))
    if mel:
      mel=np.mean(librosa.feature.melspectrogram (X, sr=sample_rate). T, axis=0)
      result=np.hstack((result, mel))
    return result


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))

        audFile = f'./static/{file.filename}'
        f.append(f'./static/{file.filename}')
        emotions={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
        em={0:'neutral',1:'calm',2:'happy',3:'sad',4:'angry',5:'fearful',6:'disgust',7:'surprised'}
        x =[]
        feature=extract_feature(audFile, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        x = np.array(x)
        predictions = np.argmax(model.predict(x),axis=-1)
        n=predictions[0]
        res = em[n]
    return render_template('index.html', prediction_text=str(res))


@app.route('/play',methods=['POST'])
def play():
    ROOT_DIR = os.path.abspath(f[-1])
    root = str(ROOT_DIR)
    root = root.replace("\\","\\\\")
    print(root)
    aud= root
    if request.method == 'POST':
        audio = playsound(aud , True)
    return render_template('index.html', sound=audio)
    

@app.route('/nextpage')
def nextpage():
    return render_template('main.html')
 

if __name__ == "__main__":
    app.debug = True
    app.run()

