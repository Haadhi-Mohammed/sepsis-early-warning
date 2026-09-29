"""
Foundation stack: the long-lived pieces every later phase builds on.

  - Cost guardrail: a monthly AWS Budget that emails before money is spent
  - Storage: one S3 bucket for data, one for model artifacts
  - SageMaker execution role: what the pipeline and its jobs run as
  - Model Package Group: the Model Registry entry every trained model
    version is recorded in
  - GitHub Actions role: lets CI deploy through OIDC, with no stored keys
"""

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from cdk_nag import NagSuppressions
from constructs import Construct

GITHUB_OIDC_URL = 'https://token.actions.githubusercontent.com'
MODEL_PACKAGE_GROUP = 'sepsis-warning'

# AWS-owned ECR registries that host the prebuilt SageMaker containers we
# use instead of building our own images. The account differs per region
# for some frameworks (from sagemaker.core.image_uris.retrieve).
SAGEMAKER_IMAGE_REGISTRIES = {
    'ap-south-1': {
        'deep-learning-containers': '763104351884',   # PyTorch training/inference
        'sklearn-processing':       '720646828776',   # SKLearn processing
    },
    'us-east-1': {
        'deep-learning-containers': '763104351884',
        'sklearn-processing':       '683313688378',
    },
}


class FoundationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, *,
                 github_repo: str, budget_email: str,
                 monthly_budget_usd: int, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._budget(budget_email, monthly_budget_usd)

        self.data_bucket = self._bucket(
            'DataBucket', 'Raw PhysioNet files and processed train/val/test splits')
        self.artifacts_bucket = self._bucket(
            'ArtifactsBucket', 'Model bundles, evaluation reports, pipeline outputs')

        self.model_group = sagemaker.CfnModelPackageGroup(
            self, 'ModelPackageGroup',
            model_package_group_name=MODEL_PACKAGE_GROUP,
            model_package_group_description=(
                'Sepsis early warning LSTM. Versions are registered by the '
                'training pipeline as PendingManualApproval; approving one '
                'releases it for deployment.'),
        )

        self.sagemaker_role = self._sagemaker_role()
        self.github_role = self._github_role(github_repo)

        # Values later phases and the CLI need. `cdk deploy` prints them.
        CfnOutput(self, 'DataBucketName', value=self.data_bucket.bucket_name)
        CfnOutput(self, 'ArtifactsBucketName', value=self.artifacts_bucket.bucket_name)
        CfnOutput(self, 'ModelPackageGroupName', value=MODEL_PACKAGE_GROUP)
        CfnOutput(self, 'SageMakerRoleArn', value=self.sagemaker_role.role_arn)
        CfnOutput(self, 'GitHubDeployRoleArn', value=self.github_role.role_arn)

    # ── Cost guardrail ─────────────────────────────────────────────────────
    def _budget(self, email: str, limit_usd: int) -> None:
        """
        Emails at 50% and 100% of actual spend, and when AWS *forecasts*
        the month will exceed the limit (an early warning before it happens).

        include_credit=False matters on an account with promotional credits:
        credits would otherwise cancel out the cost, the budget would always
        read ~$0, and the alerts would never fire.
        """
        def alert(kind: str, percent: int):
            return budgets.CfnBudget.NotificationWithSubscribersProperty(
                notification=budgets.CfnBudget.NotificationProperty(
                    notification_type=kind,             # ACTUAL or FORECASTED
                    comparison_operator='GREATER_THAN',
                    threshold=percent,
                    threshold_type='PERCENTAGE',
                ),
                subscribers=[budgets.CfnBudget.SubscriberProperty(
                    subscription_type='EMAIL', address=email)],
            )

        budgets.CfnBudget(
            self, 'MonthlyBudget',
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name='sepsis-warning-monthly',
                budget_type='COST',
                time_unit='MONTHLY',
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=limit_usd, unit='USD'),
                cost_types=budgets.CfnBudget.CostTypesProperty(
                    include_credit=False, include_refund=False),
            ),
            notifications_with_subscribers=[
                alert('ACTUAL', 50),
                alert('ACTUAL', 100),
                alert('FORECASTED', 100),
            ],
        )

    # ── Storage ────────────────────────────────────────────────────────────
    def _bucket(self, construct_id: str, purpose: str) -> s3.Bucket:
        """
        Private, encrypted, HTTPS-only bucket.

        Names are left for CDK to generate (unique, no collisions with the
        global S3 namespace); the real names are exported as stack outputs.
        Everything in these buckets can be regenerated (raw data is on disk,
        models are retrainable), so they are deleted with the stack rather
        than left behind costing money.
        """
        bucket = s3.Bucket(
            self, construct_id,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[s3.LifecycleRule(
                abort_incomplete_multipart_upload_after=Duration.days(7))],
        )
        NagSuppressions.add_resource_suppressions(bucket, [{
            'id': 'AwsSolutions-S1',
            'reason': f'{purpose}. Single-user portfolio project: CloudTrail '
                      'covers who-did-what; per-request access logs would add '
                      'cost and a third bucket with no reader.',
        }])
        return bucket

    # ── SageMaker execution role ───────────────────────────────────────────
    def _sagemaker_role(self) -> iam.Role:
        """
        The identity the training pipeline runs as, and that its jobs run
        as: data access, logging, starting its own jobs, and registering
        model versions in this project's Model Package Group only.

        The aws:SourceAccount condition stops SageMaker from assuming this
        role on behalf of a *different* account (the "confused deputy"
        problem).
        """
        role = iam.Role(
            self, 'SageMakerExecutionRole',
            description='Runs SageMaker processing, training and inference for sepsis-warning',
            assumed_by=iam.ServicePrincipal(
                'sagemaker.amazonaws.com',
                conditions={'StringEquals': {'aws:SourceAccount': self.account}}),
        )
        self.data_bucket.grant_read_write(role)
        self.artifacts_bucket.grant_read_write(role)

        role.add_to_policy(iam.PolicyStatement(
            sid='JobLogs',
            actions=['logs:CreateLogGroup', 'logs:CreateLogStream',
                     'logs:PutLogEvents', 'logs:DescribeLogStreams'],
            resources=[f'arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*'],
        ))
        role.add_to_policy(iam.PolicyStatement(
            sid='JobMetrics',
            actions=['cloudwatch:PutMetricData'],
            resources=['*'],   # PutMetricData has no resource ARNs; scoped by namespace
            conditions={'StringLike': {'cloudwatch:namespace': '/aws/sagemaker/*'}},
        ))
        role.add_to_policy(iam.PolicyStatement(
            sid='PullPrebuiltSageMakerImages',
            actions=['ecr:BatchGetImage', 'ecr:GetDownloadUrlForLayer',
                     'ecr:BatchCheckLayerAvailability'],
            resources=[f'arn:aws:ecr:{self.region}:{acct}:repository/*'
                       for acct in SAGEMAKER_IMAGE_REGISTRIES[self.region].values()],
        ))
        role.add_to_policy(iam.PolicyStatement(
            sid='EcrLogin',
            actions=['ecr:GetAuthorizationToken'],
            resources=['*'],   # this action only supports "*"
        ))

        # ── Running the pipeline ──
        # A SageMaker Pipeline doesn't do work itself: it calls
        # CreateProcessingJob / CreateTrainingJob as this role, then polls
        # them with Describe*.
        sm = f'arn:aws:sagemaker:{self.region}:{self.account}'
        role.add_to_policy(iam.PolicyStatement(
            sid='RunPipelineJobs',
            actions=['sagemaker:CreateProcessingJob', 'sagemaker:DescribeProcessingJob',
                     'sagemaker:StopProcessingJob',
                     'sagemaker:CreateTrainingJob', 'sagemaker:DescribeTrainingJob',
                     'sagemaker:StopTrainingJob',
                     'sagemaker:AddTags'],
            resources=[f'{sm}:processing-job/*', f'{sm}:training-job/*'],
        ))
        role.add_to_policy(iam.PolicyStatement(
            sid='RegisterModelVersions',
            actions=['sagemaker:CreateModelPackage', 'sagemaker:DescribeModelPackage',
                     'sagemaker:DescribeModelPackageGroup', 'sagemaker:AddTags'],
            resources=[f'{sm}:model-package-group/{MODEL_PACKAGE_GROUP}',
                       f'{sm}:model-package/{MODEL_PACKAGE_GROUP}/*'],
        ))
        # Each job it creates must run *as* this role, and handing a role to
        # a service requires iam:PassRole. Limiting it to itself and to
        # SageMaker means the pipeline can't hand out any more powerful role.
        role.add_to_policy(iam.PolicyStatement(
            sid='PassSelfToSageMakerJobs',
            actions=['iam:PassRole'],
            resources=[role.role_arn],
            conditions={'StringEquals': {'iam:PassedToService': 'sagemaker.amazonaws.com'}},
        ))

        NagSuppressions.add_resource_suppressions(role, [{
            'id': 'AwsSolutions-IAM5',
            'reason': 'Wildcards are scoped: bucket objects (bucket/*) of this '
                      "project's buckets only, SageMaker's own log groups, AWS's "
                      'prebuilt-image registries, this account\'s processing/'
                      'training jobs (job names are generated per run), versions '
                      'of this project\'s model group only, and two actions that '
                      'only accept "*" (PutMetricData, limited by namespace, and '
                      'GetAuthorizationToken).',
        }], apply_to_children=True)
        return role

    # ── GitHub Actions deploy role (OIDC) ──────────────────────────────────
    def _github_role(self, repo: str) -> iam.Role:
        """
        GitHub Actions gets short-lived AWS credentials by presenting a
        signed OIDC token that says which repo and branch the job is from.
        AWS checks the token against this trust policy, so no access keys
        are stored in GitHub secrets.

        Only workflows on the main branch of this one repository can assume
        the role, and all it can do is hand over to the CDK bootstrap roles,
        which perform the actual deployment.
        """
        provider = iam.OidcProviderNative(
            self, 'GitHubOidcProvider',
            url=GITHUB_OIDC_URL,
            client_ids=['sts.amazonaws.com'],
        )
        role = iam.Role(
            self, 'GitHubDeployRole',
            description=f'Assumed by GitHub Actions on {repo}@main to run cdk deploy',
            max_session_duration=Duration.hours(1),
            assumed_by=iam.WebIdentityPrincipal(
                provider.oidc_provider_arn,
                conditions={'StringEquals': {
                    'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
                    'token.actions.githubusercontent.com:sub': f'repo:{repo}:ref:refs/heads/main',
                }},
            ),
        )
        role.add_to_policy(iam.PolicyStatement(
            sid='UseCdkBootstrapRoles',
            actions=['sts:AssumeRole'],
            resources=[f'arn:aws:iam::{self.account}:role/cdk-hnb659fds-*-{self.account}-{self.region}'],
        ))
        NagSuppressions.add_resource_suppressions(role, [{
            'id': 'AwsSolutions-IAM5',
            'reason': 'Wildcard matches only the four CDK bootstrap roles '
                      '(deploy, lookup, file- and image-publishing) of this '
                      'account and region.',
        }], apply_to_children=True)
        return role
