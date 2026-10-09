"""
Serving stack: the approved model behind a SageMaker Serverless endpoint,
with a public HTTPS API in front.

    Function URL (HTTPS) ─► Lambda "api" ─► Endpoint "sepsis-warning"
                                              └─ Endpoint config (serverless)
                                                   └─ Model (PyTorch inference
                                                      container + model.tar.gz)

A Lambda Function URL rather than API Gateway: a serverless cold start takes
~70 s, beyond API Gateway HTTP APIs' 30 s limit, while a Lambda can wait up to
15 minutes.

Which model is served comes from cdk.json ("serving": model version + code
hash), written by `python -m serving.package_model`. Changing it replaces
the Model and endpoint config, and CloudFormation updates the endpoint to
the new config in place (same name, same URL).
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from cdk_nag import NagSuppressions
from constructs import Construct

ENDPOINT_NAME = 'sepsis-warning'

# AWS's prebuilt PyTorch inference container, matching the training image
# version used by the pipeline (sagemaker.core.image_uris.retrieve).
INFERENCE_IMAGE = '763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.5-cpu-py311'


class ServingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, *,
                 artifacts_bucket: s3.IBucket, sagemaker_role: iam.IRole,
                 model_version: int, code_version: str,
                 memory_mb: int = 3072, max_concurrency: int = 2,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        model_data = (f's3://{artifacts_bucket.bucket_name}/serving/'
                      f'v{model_version}-{code_version}/model.tar.gz')

        model = sagemaker.CfnModel(
            self, 'Model',
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=INFERENCE_IMAGE.format(region=self.region),
                model_data_url=model_data,
                environment={
                    # Tells the container which file in code/ handles requests
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': self.region,
                },
            ),
            tags=[{'key': 'model-version', 'value': str(model_version)},
                  {'key': 'code-version', 'value': code_version}],
        )

        config = sagemaker.CfnEndpointConfig(
            self, 'EndpointConfig',
            production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                variant_name='AllTraffic',
                model_name=model.attr_model_name,
                # Serverless: no instances to manage or pay for while idle.
                # Memory also sets the CPU share; max_concurrency caps cost.
                serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                    memory_size_in_mb=memory_mb,
                    max_concurrency=max_concurrency,
                ),
            )],
        )

        self.endpoint = sagemaker.CfnEndpoint(
            self, 'Endpoint',
            endpoint_name=ENDPOINT_NAME,
            endpoint_config_name=config.attr_endpoint_config_name,
        )

        self.function_url = self._api(served_model=f'v{model_version} (code {code_version})')

        CfnOutput(self, 'EndpointName', value=ENDPOINT_NAME)
        CfnOutput(self, 'ServedModel', value=f'v{model_version} (code {code_version})')
        CfnOutput(self, 'ApiUrl', value=self.function_url.url,
                  description='Public HTTPS API: GET /health, POST /predict')

        NagSuppressions.add_resource_suppressions(config, [
            {'id': 'AwsSolutions-SM2',
             'reason': 'Serverless endpoints have no instance storage volume to '
                       'encrypt with a customer KMS key; model artifacts are in '
                       'an S3-managed-encryption bucket.'},
        ], apply_to_children=True)
        NagSuppressions.add_resource_suppressions(model, [
            {'id': 'AwsSolutions-SM1',
             'reason': 'Serverless inference does not run inside a customer VPC; '
                       'the model only reads its artifact from S3 and the public '
                       'de-identified dataset involves no private network data.'},
        ], apply_to_children=True)

    # ── Public API ─────────────────────────────────────────────────────────
    def _api(self, served_model: str) -> lambda_.FunctionUrl:
        """
        A small Lambda that validates requests and forwards them to the
        endpoint, exposed through a Function URL (a built-in HTTPS address).

        Cost control: the endpoint's max_concurrency caps how many requests
        run at once. (Reserved Lambda concurrency isn't possible while the
        account's total Lambda limit is 10.)
        """
        log_group = logs.LogGroup(
            self, 'ApiLogs', retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY)
        fn = lambda_.Function(
            self, 'Api',
            description='Public sepsis prediction API in front of the SageMaker endpoint',
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler='handler.handler',
            code=lambda_.Code.from_asset(str(Path(__file__).parent.parent / 'serving' / 'api_lambda')),
            memory_size=256,
            # Must outlast a serverless cold start (~70 s)
            timeout=Duration.seconds(150),
            environment={'ENDPOINT_NAME': ENDPOINT_NAME, 'SERVED_MODEL': served_model},
            log_group=log_group,
        )
        fn.add_to_role_policy(iam.PolicyStatement(
            sid='InvokeSepsisEndpoint',
            actions=['sagemaker:InvokeEndpoint'],
            resources=[f'arn:{self.partition}:sagemaker:{self.region}:{self.account}'
                       f':endpoint/{ENDPOINT_NAME}'],
        ))
        url = fn.add_function_url(
            # Public, like the current demo API; the model sees no personal data
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors=lambda_.FunctionUrlCorsOptions(
                allowed_origins=['*'],
                allowed_methods=[lambda_.HttpMethod.GET, lambda_.HttpMethod.POST],
                allowed_headers=['content-type'],
            ),
        )
        NagSuppressions.add_resource_suppressions(fn, [
            {'id': 'AwsSolutions-IAM4',
             'reason': 'AWSLambdaBasicExecutionRole only allows writing this '
                       "function's own CloudWatch logs.",
             'appliesTo': ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']},
        ], apply_to_children=True)
        return url
