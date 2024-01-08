from keras.layers import Embedding
from keras.utils import pad_sequences
from keras.preprocessing.text import one_hot
from flask import Flask , render_template  , request
from News_Classification.logger import logging
from News_Classification.exception import AppException
from keras.models import load_model
import sys
import pickle
import numpy as np
import os
from flask_cors import CORS, cross_origin
from News_Classification.constant.training_pipeline import VOC_SIZE , ARTIFACT_DIR
from News_Classification.entity.config_entity import ModelTrainingConfig
from News_Classification.pipeline.training_pipeline import TrainPipeline


app = Flask (__name__)
CORS(app)
max_sentence_length : int = 10030






@app.route('/train')
def train():

    obj = TrainPipeline()
    result = obj.run_pipeline()
    return render_template('train.html' , result =  result)


@app.route('/')
def index():

    return render_template('index.html')

@app.route('/submit' , methods=['POST'])
def submit():

    try:
        print('try:')
        user_input = request.form['user_input']
        print(user_input)
        logging.info(user_input)
        onehot_list = []
        onehot_repr = list(one_hot(user_input , VOC_SIZE))
        print(onehot_repr)
        onehot_list.append(onehot_repr)
        logging.info("onehot encoding success")
        embedded_doc = pad_sequences(onehot_list , padding='pre' , maxlen=max_sentence_length)
        logging.info("Embeddding success")

        model_path = ModelTrainingConfig.model_file_path
        logging.info(model_path)     



        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logging.info("Model load Successfull")

        with open(os.path.join(ARTIFACT_DIR , 'encoder.pkl') , 'rb') as f:
            encoder = pickle.load(f)

        logging.info("Encoder load Successfull")
                
        prediction = model.predict(embedded_doc)

        category = encoder.inverse_transform([np.argmax(prediction)])


        return render_template('index.html' , result=category)

    
    except Exception as e:

        raise AppException(e, sys)
    

if __name__ == '__main__':    
    print("app.py running>>>")
    app.run(host="0.0.0.0", port=8080)