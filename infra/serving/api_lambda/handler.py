"""
Public HTTPS API for the sepsis model (AWS Lambda behind a Function URL).

Same routes and JSON as the FastAPI service, so clients only change the URL:
    GET  /health   -> {"status": "healthy", ...}  (does not wake the model)
    POST /predict  -> forwards the request to the SageMaker endpoint

The first request after the endpoint has been idle waits for a serverless
cold start (~70 s); the function timeout allows for it.
"""

import base64
import json
import os

import boto3
from botocore.config import Config

ENDPOINT = os.environ['ENDPOINT_NAME']
SERVED_MODEL = os.environ.get('SERVED_MODEL', '')
MAX_HOURS = 336

# Long read timeout and no automatic retries: a cold start is slow but succeeds,
# and retrying would only queue a second cold invocation.
runtime = boto3.client('sagemaker-runtime',
                       config=Config(read_timeout=150, retries={'max_attempts': 0}))


def respond(status: int, body: dict) -> dict:
    return {'statusCode': status,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(body)}


def handler(event, context):
    method = event['requestContext']['http']['method']
    path = event.get('rawPath', '/').rstrip('/') or '/'

    if method == 'GET' and path in ('/', '/health'):
        return respond(200, {'status': 'healthy', 'endpoint': ENDPOINT,
                             'served_model': SERVED_MODEL})

    if method != 'POST' or path != '/predict':
        return respond(404, {'detail': f'No route for {method} {path}'})

    raw = event.get('body') or ''
    if event.get('isBase64Encoded'):
        raw = base64.b64decode(raw).decode()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError:
        return respond(422, {'detail': 'Body must be JSON'})
    readings = request.get('readings')
    if not isinstance(readings, list) or not 1 <= len(readings) <= MAX_HOURS:
        return respond(422, {'detail': f'"readings" must be a list of 1 to {MAX_HOURS} hourly readings'})

    try:
        result = runtime.invoke_endpoint(
            EndpointName=ENDPOINT, ContentType='application/json',
            Accept='application/json', Body=json.dumps(request))
    except runtime.exceptions.ModelError as e:
        # The model container rejected or failed on the input
        print(f'ModelError: {e}')
        return respond(422, {'detail': 'The model could not process these readings'})
    except runtime.exceptions.ThrottlingException:
        return respond(429, {'detail': 'Busy, please retry in a few seconds'})
    return respond(200, json.loads(result['Body'].read()))
