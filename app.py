from __future__ import division, print_function
import numpy as np
import pandas as pd
import pickle
import joblib
import os

from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer


import sys
import glob
import re


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, model_from_json
from keras.preprocessing import image

# Flask utils
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/EmpChurnPredict")
def empchurn():
    return render_template("emp_churn1.html")

@app.route("/HeartStrokePredict")
def heartstrokepredict():
    return render_template("heart_stroke.html")


@app.route("/BrainTumorPredict")
def brainTumorPredict():
    return render_template("brain_tumor_predict.html")

@app.route("/MalariaPredict")
def malariaPredict():
    return render_template("malaria_predict.html")


@app.route("/EmailSpamClassify")
def emailSpamClassify():
    return render_template("email_spam_classify.html")


######## Model Evaluation for Heart Stroke Prediction #####################
@app.route("/heartStrokeResult",methods=['POST','GET'])
def heartStrokeResult():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    scaler_path = os.path.join('', 'models/heartstroke_scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)

    model_path = os.path.join('', 'models/heartstorke_dt.sav')
    dt = joblib.load(model_path)

    Y_pred = dt.predict(x)

    # for No Stroke Risk
    if Y_pred == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')


######## Model Evaluation for Employee Churn Prediction #####################
@app.route("/empChurnResult",methods=['POST','GET'])
def empChurnResult():
    satisfaction_level=float(request.form['satisfaction'])
    number_project=int(request.form['numberOfProjects'])
    average_montly_hours=int(request.form['avgMonthlyHours'])
    time_spend_company = int(request.form['timeSpendCompany'])
    promotion_last_5years = int(request.form['promotionLast5Years'])
    salary = str(request.form['salary'])

    if salary == 'Low':
        salary_high = 0
        salary_low = 1
        salary_medium = 0

    elif salary == 'Medium':
        salary_high = 0
        salary_low = 0
        salary_medium = 1

    else:
        salary_high = 1
        salary_low = 0
        salary_medium = 0


    x=np.array([satisfaction_level,number_project,average_montly_hours,time_spend_company,promotion_last_5years,
                salary_high,salary_low,salary_medium]).reshape(1,-1)


    model = pickle.load(open('models/empatr_rf_model.pkl', 'rb'))

    Y_pred=model.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('empretain.html')
    else:
        return render_template('empleft.html')




######## Model Evaluation for Brain Tumor Detection #####################

def get_brTumorModel():
    global brTumor_model
    #model = load_model('.h5')

    #json_file = open('models/brtumor_model3.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()

    #brTumor_model = model_from_json(loaded_model_json)

    #brTumor_model.load_weights("models/brtumor_model3.h5")

    #brTumor_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    brTumor_model = load_model('models/brtumor1_vgg19.h5')

    print('model loaded')

def brTumor_model_predict(img_path, model):

    print('inside model_predict function')
    #img = image.load_img(img_path, target_size=(180, 180))
    img = image.load_img(img_path, target_size=(50, 50))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    img_data = x/255

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    #x = preprocess_input(x)


    preds = model.predict(img_data)
    print('preds in brTumor_model_predict')
    print(preds)
    return np.argmax(preds, axis=1)

@app.route('/brainTumorPredictResult', methods=['GET', 'POST'])
def brainTumorPredictResult():
    if request.method == 'POST':
        get_brTumorModel()
        # Get the file from post request
        print('inside upload function')
        f = request.files['file']

        print(f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)

        #tmp_filepath = "C:\Rajiv\work\DataScience\BrainCancer_Dataset\Brain Tumor Data Set\htn1.jpg"
        #tmp_filepath = "C:\\Rajiv\\work\\DataScience\\BrainCancer_Dataset\\Brain Tumor Data Set\\htn1.jpg"
        #print(tmp_filepath)

        #file_path2 = os.path.join(os.path.dirname(__file__), 'testimages', f.filename)
        #print(file_path2)

        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)

        #loaded_model.load_weights("model.h5")

        # Make prediction
        preds = brTumor_model_predict(file_path, brTumor_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        print('Predicted Values in result function: ')
        print(preds)
        #print(np.argmax(preds))
        label_list = ['Healthy', 'Tumor']
        #result = label_list[np.argmax(preds)]
        result = label_list[np.asscalar(preds)]
        #result = label_list[preds]
        print(result)

        return result
    return None


######## Model Evaluation for Malaria Prediction #####################

def get_malariaModel():
    global malaria_model
    #model = load_model('.h5')

    #json_file = open('models/brtumor_model3.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #brTumor_model = model_from_json(loaded_model_json)

    #malaria_model.load_weights("models/malaria_prediction_model.h5")
    malaria_model=load_model("models/malaria_prediction_model.h5")

    #brTumor_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('malaria model loaded')

def malaria_model_predict(img_path, model):

    print('inside malaria model_predict function')
    #img = image.load_img(img_path, target_size=(180, 180))
    #img = image.load_img(img_path, target_size=(50, 50, 3))
    data = image.load_img(img_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    # Preprocessing the image
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    #x = preprocess_input(x)


    preds = model.predict(data)
    print('preds in malaria_model_predict')
    print(preds)
    return np.argmax(preds, axis=1)

@app.route('/malariaPredictResult', methods=['GET', 'POST'])
def malariaPredictResult():
    if request.method == 'POST':
        get_malariaModel()
        # Get the file from post request
        print('inside malariaPredictResult upload function')
        f = request.files['file']

        print(f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)

        #tmp_filepath = "C:\Rajiv\work\DataScience\BrainCancer_Dataset\Brain Tumor Data Set\htn1.jpg"
        #tmp_filepath = "C:\\Rajiv\\work\\DataScience\\BrainCancer_Dataset\\Brain Tumor Data Set\\htn1.jpg"
        #print(tmp_filepath)

        #file_path2 = os.path.join(os.path.dirname(__file__), 'testimages', f.filename)
        #print(file_path2)

        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)

        #loaded_model.load_weights("model.h5")

        # Make prediction
        preds = malaria_model_predict(file_path, malaria_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        print('Predicted Malaria result Values in result function: ')
        print(preds)
        #print(np.argmax(preds))
        #label_list = ['Healthy', 'Tumor']
        #result = label_list[np.argmax(preds)]

        #indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}

        #predicted_class = np.asscalar(np.argmax(preds, axis=1))
        #accuracy = round(result[0][predicted_class] * 100, 2)
        indices = ['PARASITIC', 'Uninfected']
        result = indices[np.asscalar(preds)]


        #result = label_list[preds]
        print(result)

        return result
    return None



######## Model Evaluation for Email Spam classification #####################
@app.route("/emailSpamClassifyresult", methods=['POST','GET'])
def emailSpamClassifyresult():
    print('Loading the model...')
    model = pickle.load(open('models/spamclassify.pkl', 'rb'))
    count_vect = pickle.load(open('models/spamclassify_cv', 'rb'))
    print('Model is loaded')

    email_text = str(request.form['emailtext'])
    print('email_text: ' + email_text)

    corpus = []
    corpus.append(email_text)
    print('corpus:: ')
    print(corpus)

    print('start prediction ...')
    test_result = model.predict(count_vect.transform(corpus))
    print('test result :: ')
    print(test_result)

    if test_result==0:
        return render_template('noSpam.html')
    else:
        return render_template('spam.html')




if __name__ == '__main__':
    app.run(debug=True, port=5567)