import os
import sys
import zipfile
import gdown
from News_Classification.logger import logging
from News_Classification.exception import AppException
from News_Classification.entity.config_entity import DataIngestionConfig
from News_Classification.entity.artifacts_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self,data_Ingestion_Config : DataIngestionConfig = DataIngestionConfig()):

        try:
            self.Data_Ingestion_Config = data_Ingestion_Config
        except Exception as e:
            raise AppException(e , sys)
        

    def download_data(self) -> str :

        try:

            dataset_url = self.Data_Ingestion_Config.data_download_url
            zip_download_dir = self.Data_Ingestion_Config.data_ingestion_dir


            os.makedirs(zip_download_dir , exist_ok=True)

            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir , data_file_name)
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")

            file_id = dataset_url.split("/")[-2]
            prefix  ='https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id , zip_file_path)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")
 
            return zip_file_path
            
        except Exception as e:
            raise AppException(e , sys)
        
    
    def extract_zip_file(self , Zip_file_path : str) -> str:

        try:

            feature_store_path = self.Data_Ingestion_Config.feature_store_file_path
            os.makedirs(feature_store_path , exist_ok=True)

            with zipfile.ZipFile(Zip_file_path , 'r') as zipref:
                zipref.extractall(feature_store_path)

            logging.info(f"Extracting zip file: {Zip_file_path} into dir: {feature_store_path}")

            return feature_store_path


        except Exception as e:
            raise AppException(e , sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:

            zip_file_path  = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)

            Data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path= zip_file_path,
                feature_store_path=feature_store_path
            )

            return Data_ingestion_artifact

        except Exception as e:

            raise AppException(e , sys)










    