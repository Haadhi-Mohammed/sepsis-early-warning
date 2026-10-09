#!/usr/bin/env python3
"""
CDK app for the sepsis early warning system.

Run from infra/ with the virtualenv active:
    npx cdk synth      # build the CloudFormation template
    npx cdk diff       # compare with what is deployed
    npx cdk deploy     # deploy

Required environment variable:
    BUDGET_EMAIL   address for AWS Budgets alerts (kept out of the repo)
"""

import os

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from sepsis_infra.foundation_stack import FoundationStack
from sepsis_infra.serving_stack import ServingStack
from sepsis_infra.web_stack import WebStack

app = cdk.App()

# Explicit account + region: CDK resolves the account from your AWS
# credentials (CDK_DEFAULT_ACCOUNT) and we pin the region to the one chosen
# for the project, so a stray AWS_REGION can't deploy somewhere unexpected.
# Mumbai: ~50 ms from the developer in India vs ~300 ms to us-east-1, for
# ~5% higher SageMaker prices (cents at this scale), and data stays in India.
env = cdk.Environment(account=os.environ.get('CDK_DEFAULT_ACCOUNT'),
                      region='ap-south-1')

budget_email = os.environ.get('BUDGET_EMAIL')
if not budget_email:
    raise SystemExit('Set BUDGET_EMAIL to the address for AWS cost alerts')

foundation = FoundationStack(
    app, 'SepsisFoundation',
    env=env,
    github_repo=app.node.get_context('github_repo'),
    budget_email=budget_email,
    monthly_budget_usd=int(app.node.get_context('monthly_budget_usd')),
    model_registry=str(app.node.get_context('model_registry')).lower() == 'true',
    # Optional: SageMaker Studio's execution role ("path/name"), to let Studio
    # show model evaluation reports. Leave unset if Studio isn't used.
    studio_role=app.node.try_get_context('studio_role'),
    description='Sepsis early warning: storage, IAM roles, CI access and cost guardrails',
    # CloudFormation refuses to delete a protected stack until protection
    # is switched off, so a mistyped `cdk destroy` can't wipe the data.
    termination_protection=True,
)

# Serving: deployed once a model version has been packaged for serving
# (python -m serving.package_model writes "serving" into cdk.json).
serving = app.node.try_get_context('serving')
if serving:
    serving_stack = ServingStack(
        app, 'SepsisServing',
        env=env,
        artifacts_bucket=foundation.artifacts_bucket,
        sagemaker_role=foundation.sagemaker_role,
        model_version=int(serving['model_version']),
        code_version=serving['code_version'],
        description='Sepsis early warning: approved model on a SageMaker Serverless endpoint',
    )

    # Dashboard: static site on S3 + CloudFront, with /api/* forwarded to the
    # serving stack's API
    WebStack(
        app, 'SepsisWeb',
        env=env,
        api_function_url=serving_stack.function_url,
        description='Sepsis early warning: dashboard on S3 + CloudFront',
    )

# Every resource gets these tags, so costs can be filtered per project in
# Cost Explorer (after activating the tag in Billing > Cost allocation tags).
cdk.Tags.of(app).add('project', 'sepsis-warning')
cdk.Tags.of(app).add('managed-by', 'cdk')

# cdk-nag checks the synthesised template against AWS Solutions security
# rules and fails synth on violations, like a linter for infrastructure.
cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

app.synth()
