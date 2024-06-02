import os
import logging
from pathlib import Path
import streamlit as st
import pickle
import boto3
import pandas as pd
import numpy as np
import yaml
import sklearn 


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load configuration file
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration file loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        st.error("Error loading configuration file. Please check the logs.")
        raise

# Load the configuration
config = load_config('config.yaml')

# AWS Configuration
BUCKET_NAME = config['aws']['bucket_name']
MODEL_FILE_KEYS = config['aws']['model_file_keys']
DATA_FILE_KEY = config['aws']['data_file_key']
AWS_ACCESS_KEY_ID = os.getenv('aws_access_key_id')
AWS_SECRET_ACCESS_KEY = os.getenv('aws_secret_access_key')
AWS_REGION_NAME = config['aws']['region_name']
MODEL_STORE='models/'

os.makedirs(MODEL_STORE, exist_ok=True)

# Initialize S3 client with credentials
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME
    )
    logger.info("S3 client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing S3 client: {e}")
    st.error("Error initializing S3 client. Please check the logs.")
    raise

# Function to download a file from S3 and return a BytesIO object
def download_file_from_s3(bucket, key):
    try:
        model_path = os.path.join(MODEL_STORE, os.path.basename(key))
        s3.download_file(bucket, key, model_path)
        logger.info(f"File {key} downloaded from S3 successfully")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        st.error("Error downloading file from S3. Please check the logs.")
        raise

# Load the model from S3
def load_model_from_s3(bucket, key):
    path = download_file_from_s3(bucket, key)
    try:
        model = pickle.load(open(path, 'rb'))
        logger.info(f"Model {key} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Error loading model. Please check the logs.")
        raise

# Load the data from S3
def load_data_from_s3(bucket, key):
    data = download_file_from_s3(bucket, key)
    try:
        df = pd.read_csv(data)
        logger.info(f"Data {key} loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Error loading data. Please check the logs.")
        raise

# Load data
data = load_data_from_s3(BUCKET_NAME, DATA_FILE_KEY)

# Streamlit application layout
st.title("Clouds Data Prediction")
st.write("This app predicts the class of cloud data using a pre-trained model.")

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ["Model 1", "Model 2"])
model_file_key = MODEL_FILE_KEYS["model1"] if model_choice == "Model 1" else MODEL_FILE_KEYS["model2"]

# Load the selected model
model = load_model_from_s3(BUCKET_NAME, model_file_key)

# Sidebar inputs for feature values
st.sidebar.header("Input Parameters")

# Assuming your data has columns 'log_entropy', 'IR_norm_range', and 'entropy_x_contrast'
try:
    log_entropy = st.sidebar.slider("Log Entropy", float(data['log_entropy'].min()), float(data['log_entropy'].max()), float(data['log_entropy'].mean()))
    IR_norm_range = st.sidebar.slider("IR Norm Range", float(data['IR_norm_range'].min()), float(data['IR_norm_range'].max()), float(data['IR_norm_range'].mean()))
    entropy_x_contrast = st.sidebar.slider("Entropy x Contrast", float(data['entropy_x_contrast'].min()), float(data['entropy_x_contrast'].max()), float(data['entropy_x_contrast'].mean()))
except KeyError as e:
    logger.error(f"Missing feature in data: {e}")
    st.error("Missing feature in data. Please check the logs.")
    raise

# Prepare input data for prediction
input_data = pd.DataFrame({
    'log_entropy': [log_entropy],
    'IR_norm_range': [IR_norm_range],
    'entropy_x_contrast': [entropy_x_contrast]
})

# Prediction section
st.subheader("Prediction")
try:
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    st.write(f"Predicted Class: {predicted_class}")

    # Optionally display prediction probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        class_index = model.classes_.tolist().index(predicted_class)
        st.write(f"Probability: {probability[class_index]:.2f}")
except Exception as e:
    logger.error(f"Error during prediction: {e}")
    st.error("Error during prediction. Please check the logs.")
    raise

# Handling errors and logging
try:
    # Add your logging and error handling code here
    pass
except Exception as e:
    st.error(f"An error occurred: {e}")
    logger.error(f"An unexpected error occurred: {e}")
