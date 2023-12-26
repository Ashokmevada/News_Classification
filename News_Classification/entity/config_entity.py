import os
from dataclasses import dataclass
from datetime import datetime
from News_Classification.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACT_DIR

trainingPipelineConfig : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        trainingPipelineConfig.artifacts_dir , DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path : str = os.path.join(
        data_ingestion_dir , DATA_INGESTION_FEATURE_STORE_DIR
    )

    data_download_url = DATA_DOWNLOAD_URL

@dataclass
class DataTranformationConfig:
    
    data_transformation_dir: str = DATA_TRANSFORMATION_DIR
    data_transformation_artifacts_dir : str = os.path.join(ARTIFACT_DIR , data_transformation_dir)
    tranformed_file_name: str = TRANSFORMED_FILE_NAME
    data_dir:str = DATA_DIR    
    drop_columns = DROP_COLUMNS
    inplace = INPLACE
    transformed_file_path = os.path.join(data_transformation_artifacts_dir , tranformed_file_name)