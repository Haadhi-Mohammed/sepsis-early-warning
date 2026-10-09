"""
SageMaker inference handler for the sepsis model.

AWS's prebuilt PyTorch inference container unpacks model.tar.gz into
/opt/ml/model, puts code/ (this file, the sepsis package and its
requirements) on the Python path, and calls these four functions for every
request. They wrap the same SepsisPredictor the FastAPI service uses, so the
endpoint and the API answer identically.

Request:  {"patient_id": "P1", "readings": [{"HR": 95, ...}, ...]}
Response: the same JSON body as the API's /predict
"""

import json

from sepsis.inference import SepsisPredictor


def model_fn(model_dir, context=None):
    """Called once per container start: load model, preprocessor and SHAP."""
    return SepsisPredictor(model_dir)


def input_fn(body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported content type {content_type}; send application/json')
    request = json.loads(body)
    if not isinstance(request.get('readings'), list):
        raise ValueError('Request must contain a "readings" list of hourly readings')
    return request


def predict_fn(request, predictor):
    result = predictor.predict(request['readings'])
    return {'patient_id': request.get('patient_id'), **result}


def output_fn(prediction, accept='application/json'):
    return json.dumps(prediction)
