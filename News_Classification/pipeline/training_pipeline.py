import os
import sys
from News_Classification.components.data_ingestion import DataIngestion
from News_Classification.logger import logging
from News_Classification.exception import AppException
from News_Classification.entity.config_entity import DataIngestionConfig
from News_Classification.entity.artifacts_entity import DataIngestionArtifact


class TrainPipeline:

    def __init__(self):

        self.Data_Ingestion_Config = DataIngestionConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact :

        try:

            logging.info(
                    "Entered the start_data_ingestion method of TrainPipeline class"
                )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_Ingestion_Config= self.Data_Ingestion_Config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the data from URL")
            logging.info(
                    "Exited the start_data_ingestion method of TrainPipeline class"
                )
    
            return data_ingestion_artifact

        except Exception as e:

            raise AppException(e)
        
    def run_pipeline(self) -> None:

        try:
            data_ingestion_artifacts = self.start_data_ingestion()

        except Exception as e:
            raise AppException(e)