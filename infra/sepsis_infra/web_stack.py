"""
Web stack: the dashboard as a static site on S3, served through CloudFront.

    Browser ─► CloudFront ─┬─ /*      ─► S3 bucket (private; Origin Access Control)
                           └─ /api/*  ─► Lambda Function URL (prediction API)

Serving page and API from one CloudFront address means no CORS, HTTPS
everywhere, and a global CDN in front of the API (which also gives clients a
well-routed IPv6 path). The page itself has no server: it costs pennies.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from cdk_nag import NagSuppressions
from constructs import Construct

SITE_DIR = Path(__file__).parent.parent / 'web'


class WebStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, *,
                 api_function_url: lambda_.IFunctionUrl, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Private bucket: only CloudFront can read it (Origin Access Control)
        site_bucket = s3.Bucket(
            self, 'SiteBucket',
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        distribution = cloudfront.Distribution(
            self, 'Distribution',
            comment='Sepsis early warning dashboard and API',
            default_root_object='index.html',
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            # Includes edge locations in India and the rest of Asia
            price_class=cloudfront.PriceClass.PRICE_CLASS_200,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                '/api/*': cloudfront.BehaviorOptions(
                    origin=origins.FunctionUrlOrigin(
                        api_function_url,
                        # CloudFront's maximum without a quota increase. A cold
                        # start can take longer; the page then retries once.
                        read_timeout=Duration.seconds(60),
                        ip_address_type=cloudfront.OriginIpAddressType.IPV4,
                    ),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    # Forward everything except Host: the Function URL needs its own
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
        )

        # Upload the page and clear CloudFront's cache on every deploy
        s3deploy.BucketDeployment(
            self, 'SiteDeployment',
            sources=[s3deploy.Source.asset(str(SITE_DIR))],
            destination_bucket=site_bucket,
            distribution=distribution,
            distribution_paths=['/*'],
        )

        CfnOutput(self, 'DashboardUrl', value=f'https://{distribution.distribution_domain_name}')

        # BucketDeployment runs a CDK-managed helper Lambda (one per stack) that
        # copies files from the CDK assets bucket into the site bucket and
        # invalidates CloudFront. Its permissions and runtime are set by CDK.
        helper = next(c for c in self.node.children
                      if c.node.id.startswith('Custom::CDKBucketDeployment'))
        NagSuppressions.add_resource_suppressions(helper, [
            {'id': 'AwsSolutions-IAM4', 'reason': 'CDK-managed deployment helper: '
                                                  "basic execution role for its own logs."},
            {'id': 'AwsSolutions-IAM5', 'reason': 'CDK-managed deployment helper: copies objects '
                                                  'from the CDK assets bucket into this site bucket, '
                                                  'and CloudFront invalidations only accept "*".'},
            {'id': 'AwsSolutions-L1', 'reason': 'CDK-managed deployment helper; its runtime is '
                                                'pinned by the CDK library version.'},
        ], apply_to_children=True)

        NagSuppressions.add_resource_suppressions(site_bucket, [{
            'id': 'AwsSolutions-S1',
            'reason': 'Static files for a public demo page; CloudFront is the only '
                      'reader and per-request S3 access logs would add cost with no reader.',
        }], apply_to_children=True)
        NagSuppressions.add_resource_suppressions(distribution, [
            {'id': 'AwsSolutions-CFR1', 'reason': 'Public portfolio demo: no geographic restriction needed.'},
            {'id': 'AwsSolutions-CFR2', 'reason': 'AWS WAF costs a monthly fee per web ACL; the API is '
                                                  'capped by the endpoint max concurrency instead.'},
            {'id': 'AwsSolutions-CFR3', 'reason': 'Access logs would need another bucket and storage cost '
                                                  'for a demo; Lambda logs record API requests.'},
            {'id': 'AwsSolutions-CFR4', 'reason': 'Uses the default *.cloudfront.net certificate (no custom '
                                                  'domain), whose minimum TLS version cannot be raised.'},
        ], apply_to_children=True)
