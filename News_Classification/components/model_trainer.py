import pandas as pd
import numpy as np
import pickle
from keras.layers import Embedding
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM , Bidirectional
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from News_Classification.exception import AppException
from News_Classification.logger import logging
from News_Classification.constant.training_pipeline import *
from News_Classification.entity.config_entity import ModelTrainingConfig , DataTranformationConfig
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import re
import os
import sys

class ModelTrainer:

    def __init__(
            self ,
            model_training_config : ModelTrainingConfig,
            data_transformation_config: DataTranformationConfig ):
        
        self.model_training_config = model_training_config
        self.data_transformation_config = data_transformation_config
        self.full_corpus = []


    def clean_text(self , text):

        sentences = nltk.sent_tokenize(text)
        corpus = []
        stemmer = PorterStemmer()
        

        for i in range(len(sentences)):

            review = re.sub('[^a-zA-Z]' , ' ' , sentences[i])
            review = review.lower()
            review = review.split()
            review = [stemmer.stem(i) for i in review if not i in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)

        self.full_corpus.append('.'.join(corpus))

        ans = '.'.join(corpus)

        return ans

    def initiate_model_training(self):  

        try:

            logging.info("Training started...")

            df = pd.read_csv(self.data_transformation_config.transformed_file_path)
            df['Text'] = df['Text'].apply(self.clean_text)
            onehot_repr = [one_hot(words , self.model_training_config.voc_size) for words in self.full_corpus]
            sent_max_length = 0
            for i in self.full_corpus:

                if len(i) > sent_max_length:
                    sent_max_length = len(i)

            embedded_docs = pad_sequences(onehot_repr , padding = "pre" , maxlen = sent_max_length)
            X_final = np.array(embedded_docs)
            y_final = np.array(df['Category'])

            model = Sequential()
            model.add(Embedding(self.model_training_config.voc_size , self.model_training_config.embedded_vector_feature, input_length = sent_max_length))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(100)))
            model.add(Dropout(0.3))
            model.add(Dense(len(df['Category'].unique()) , activation = 'softmax'))

            model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics=['accuracy'])

            print(model.summary())

            X_train , X_val , y_train , y_val = train_test_split(X_final , y_final , test_size = 0.2 , random_state = 90)

            logging.info('Model fitting....')

            model.fit(X_train , y_train, validation_data=(X_val , y_val ) , epochs = 1 )

            logging.info('Model fitted')

            os.makedirs(self.model_training_config.model_dir_path , exist_ok=True)

            with open(os.path.join(self.model_training_config.model_dir_path , self.model_training_config.model_file_name ) , 'wb') as f:
                pickle.dump(model , f)

            logging.info(f'Model saved at {self.model_training_config.model_file_path}')
            return os.path.join(self.model_training_config.model_file_path , self.model_training_config.model_file_name)

        except Exception as e:

            raise AppException(e , sys)

        



