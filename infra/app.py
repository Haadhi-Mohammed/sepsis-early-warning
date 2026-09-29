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

FoundationStack(
    app, 'SepsisFoundation',
    env=env,
    github_repo=app.node.get_context('github_repo'),
    budget_email=budget_email,
    monthly_budget_usd=int(app.node.get_context('monthly_budget_usd')),
    description='Sepsis early warning: storage, IAM roles, CI access and cost guardrails',
    # CloudFormation refuses to delete a protected stack until protection
    # is switched off, so a mistyped `cdk destroy` can't wipe the data.
    termination_protection=True,
)

# Every resource gets these tags, so costs can be filtered per project in
# Cost Explorer (after activating the tag in Billing > Cost allocation tags).
cdk.Tags.of(app).add('project', 'sepsis-warning')
cdk.Tags.of(app).add('managed-by', 'cdk')

# cdk-nag checks the synthesised template against AWS Solutions security
# rules and fails synth on violations, like a linter for infrastructure.
cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

app.synth()
