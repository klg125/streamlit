import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import joblib
from io import BytesIO
import app

@pytest.fixture
def mock_load_config():
    with patch('app.load_config') as mock_config:
        mock_config.return_value = {
            'aws': {
                'bucket_name': 'test_bucket',
                'model_file_key': 'test_model.pkl',
                'data_file_key': 'test_data.csv',
                'aws_access_key_id': 'fake_access_key',
                'aws_secret_access_key': 'fake_secret_key',
                'region_name': 'us-east-1'
            }
        }
        yield mock_config

@pytest.fixture
def mock_download_file_from_s3():
    with patch('app.download_file_from_s3') as mock_download:
        yield mock_download

@pytest.fixture
def mock_joblib_load():
    with patch('app.joblib.load') as mock_load:
        yield mock_load

@pytest.fixture
def mock_read_csv():
    with patch('app.pd.read_csv') as mock_read:
        yield mock_read

def test_model_loading_failure(mock_load_config, mock_download_file_from_s3, mock_joblib_load):
    mock_download_file_from_s3.side_effect = Exception("S3 download failed")
    with pytest.raises(Exception, match="S3 download failed"):
        app.load_model_from_s3('test_bucket', 'test_model.pkl')

def test_data_loading_failure(mock_load_config, mock_download_file_from_s3, mock_read_csv):
    mock_download_file_from_s3.side_effect = Exception("S3 download failed")
    with pytest.raises(Exception, match="S3 download failed"):
        app.load_data_from_s3('test_bucket', 'test_data.csv')

def test_missing_feature_in_data(mock_load_config, mock_download_file_from_s3, mock_read_csv):
    mock_data = pd.DataFrame({
        'log_entropy': [1.0, 2.0, 3.0],
        'IR_norm_range': [0.1, 0.2, 0.3]
    })
    mock_download_file_from_s3.return_value = BytesIO(mock_data.to_csv(index=False).encode())
    mock_read_csv.return_value = mock_data

    data = app.load_data_from_s3('test_bucket', 'test_data.csv')

    with pytest.raises(KeyError):
        # Try to access a missing feature
        _ = data['entropy_x_contrast']

def test_input_data_errors(mock_load_config, mock_download_file_from_s3, mock_joblib_load, mock_read_csv):
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Prediction failed")
    mock_joblib_load.return_value = mock_model

    mock_data = pd.DataFrame({
        'log_entropy': [1.0, 2.0, 3.0],
        'IR_norm_range': [0.1, 0.2, 0.3],
        'entropy_x_contrast': [0.01, 0.02, 0.03]
    })
    mock_download_file_from_s3.return_value = BytesIO(mock_data.to_csv(index=False).encode())
    mock_read_csv.return_value = mock_data

    app.data = mock_data
    app.model = mock_model

    with pytest.raises(Exception, match="Prediction failed"):
        input_data = pd.DataFrame({
            'log_entropy': [1.5],
            'IR_norm_range': [0.15],
            'entropy_x_contrast': [0.015]
        })
        app.model.predict(input_data)
