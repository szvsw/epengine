"""Create the S3 bucket and other resources if they do not exist."""

import logging
import os

import boto3

s3 = boto3.client("s3")

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# list buckets
if __name__ == "__main__":
    bucket_name = "ml-for-bem"
    # create the bucket if it does not exist
    try:
        region = os.getenv("AWS_DEFAULT_REGION")
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={
                "LocationConstraint": region,  # pyright: ignore [reportArgumentType]
            },
        )
        logger.info(f"Bucket {bucket_name} created")
    except Exception as e:
        logger.info(f"Bucket {bucket_name} already exists, {e}")
