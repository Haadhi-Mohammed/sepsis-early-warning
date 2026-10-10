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

CloudWatch alarms email the alert address (through SNS) when the API or the
endpoint starts failing, and again when they recover.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from cdk_nag import NagSuppressions
from constructs import Construct

ENDPOINT_NAME = 'sepsis-warning'

# AWS's prebuilt PyTorch inference container, matching the training image
# version used by the pipeline (sagemaker.core.image_uris.retrieve).
INFERENCE_IMAGE = '763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.5-cpu-py311'


class ServingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, *,
                 artifacts_bucket: s3.IBucket, sagemaker_role: iam.IRole,
                 model_version: int, code_version: str, alert_email: str,
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

        self.function_url, api_function = self._api(
            served_model=f'v{model_version} (code {code_version})')
        self._alarms(alert_email, api_function)

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
    def _api(self, served_model: str) -> tuple[lambda_.FunctionUrl, lambda_.Function]:
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
        return url, fn

    # ── Monitoring ─────────────────────────────────────────────────────────
    def _alarms(self, email: str, api: lambda_.Function) -> None:
        """
        Two alarms, each emailing on failure and again on recovery:
          - API errors: the Lambda crashed or timed out (users saw a 5xx).
            Handled outcomes (422 bad input, 429 busy) are not errors.
          - Endpoint 5xx: the model container failed on a request.
        Missing data counts as healthy: an idle demo sends no metrics at all.
        Alarms cost $0.10/month each; SNS email is free at this volume.
        """
        topic = sns.Topic(self, 'Alerts', display_name='Sepsis warning alerts',
                          enforce_ssl=True)
        # AWS sends a confirmation email; alerts arrive once it is confirmed
        topic.add_subscription(subscriptions.EmailSubscription(email))
        notify = cw_actions.SnsAction(topic)

        def alarm(construct_id: str, metric: cloudwatch.Metric, description: str):
            a = cloudwatch.Alarm(
                self, construct_id,
                metric=metric,
                threshold=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                evaluation_periods=1,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description=description,
            )
            a.add_alarm_action(notify)
            a.add_ok_action(notify)

        five_minutes = Duration.minutes(5)
        alarm('ApiErrorsAlarm',
              api.metric_errors(period=five_minutes, statistic='Sum'),
              'Prediction API (Lambda) failed or timed out in the last 5 minutes.')
        alarm('EndpointErrorsAlarm',
              cloudwatch.Metric(
                  namespace='AWS/SageMaker', metric_name='Invocation5XXErrors',
                  dimensions_map={'EndpointName': ENDPOINT_NAME, 'VariantName': 'AllTraffic'},
                  period=five_minutes, statistic='Sum'),
              'SageMaker endpoint returned server errors in the last 5 minutes.')

        NagSuppressions.add_resource_suppressions(topic, [
            {'id': 'AwsSolutions-SNS2',
             'reason': 'Alert messages contain only alarm names and states. '
                       'CloudWatch alarms cannot publish to a topic encrypted '
                       'with the AWS-managed SNS key, and a customer KMS key '
                       'costs $1/month for no benefit here.'},
        ], apply_to_children=True)
