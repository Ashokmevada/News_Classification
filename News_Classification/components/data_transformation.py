import pandas as pd
import matplotlib.pyplot as plt
import nltk
import sys
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import re
import os
from News_Classification.logger import logging
from News_Classification.exception import AppException
from News_Classification.entity.artifacts_entity import DataIngestionArtifact , DataTransformationArtifact
from News_Classification.entity.config_entity import DataTranformationConfig
from sklearn.preprocessing import LabelEncoder
import pickle
from News_Classification.constant.training_pipeline import ARTIFACT_DIR




class DataTranformation:

    def __init__(self , data_transformation_config : DataTranformationConfig , data_ingestion_artifact : DataIngestionArtifact):

        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AppException(e , sys)
        
 

    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:

            df = pd.read_csv(os.path.join(self.data_ingestion_artifact.feature_store_path , "learn-ai-bbc" , "train.csv"))
            df.drop(self.data_transformation_config.drop_columns , axis =1 , inplace=self.data_transformation_config.inplace)
            df.drop_duplicates(inplace=self.data_transformation_config.inplace)
            encoder = LabelEncoder()
            df['Category'] = encoder.fit_transform(df['Category'])


            with open(os.path.join(ARTIFACT_DIR , 'encoder.pkl') , 'wb') as f:
                pickle.dump(encoder , f)


            os.makedirs(self.data_transformation_config.data_transformation_artifacts_dir , exist_ok=True)
            df.to_csv(self.data_transformation_config.transformed_file_path , index=False , header=True)


            data_transformation_artifacts= DataTransformationArtifact(
                transformed_data_path=self.data_transformation_config.transformed_file_path
            )


            return data_transformation_artifacts
        
        except Exception as e:

            raise AppException(e , sys)

    

