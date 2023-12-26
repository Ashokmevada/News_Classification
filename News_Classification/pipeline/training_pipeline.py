import os
import sys
from News_Classification.components.data_ingestion import DataIngestion
from News_Classification.components.data_transformation import DataTranformation
from News_Classification.logger import logging
from News_Classification.exception import AppException
from News_Classification.entity.config_entity import DataIngestionConfig , DataTranformationConfig
from News_Classification.entity.artifacts_entity import DataIngestionArtifact , DataTransformationArtifact


class TrainPipeline:

    def __init__(self):

        self.Data_Ingestion_config = DataIngestionConfig
        self.Data_Transformed_config = DataTranformationConfig


    def start_data_ingestion(self) -> DataIngestionArtifact :

        try:

            logging.info(
                    "Entered the start_data_ingestion method of TrainPipeline class"
                )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_Ingestion_Config= self.Data_Ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the data from URL")
            logging.info(
                    "Exited the start_data_ingestion method of TrainPipeline class"
                )
    
            return data_ingestion_artifact

        except Exception as e:

            raise AppException(e , sys)
        
    def start_data_transformation(self , data_ingestion_artifacts = DataIngestionArtifact ) -> DataTransformationArtifact:

        try :
        
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            
            data_transformation = DataTranformation(
                data_ingestion_artifact=data_ingestion_artifacts,
                data_transformation_config=self.Data_Transformed_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts

        except Exception as e:
            raise AppException(e, sys)


        
    def run_pipeline(self) -> None:

        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)

        except Exception as e:
            raise AppException(e , sys)