import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.constant.training_pipeline import (TARGET_COLUMN, 
                                                         DATA_TRANSFORMATION_IMPUTER_PARAMS,
                                                        )
from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact 
)
from network_security.entity.config_entity import DataTransformConfig
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.utils.main_utils.utils import save_numpy_arr_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact : DataValidationArtifact,
                data_transformation_config : DataTransformConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_data_transform_obj(cls) -> Pipeline:
        """
        Initialize a KNNImouter obj with the parameters speciied in the training_pipeline.py file
        and returns a pipeliine obj with the KNNImputer obj as the first step. 
        """
        logging.info("Entered get_data_transform_obj method")
        try:
            imputer : KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialized the imputer with params : {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor : Pipeline = Pipeline([("imputer",imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiated Data Transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            ## training dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)
            
            ## test dataframe
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)
            
            preprocessor = self.get_data_transform_obj()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            save_numpy_arr_data(
                self.data_transformation_config.transformed_train_file_path,
                array = train_arr
            )
            save_numpy_arr_data(
                self.data_transformation_config.transformed_test_file_path,
                array = test_arr
            )
            save_object(
                self.data_transformation_config.transformed_obj_file_path,
                preprocessor_obj
            )
            
            # prepraing artifacts
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_obj_file_path = self.data_transformation_config.transformed_obj_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact
             
        except Exception as e:
            raise NetworkSecurityException(e, sys) 