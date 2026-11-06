"""S3-compatible object storage (AWS S3, Cloudflare R2, MinIO in local dev)."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT") or None,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        # Default CRC checksums (boto3 >= 1.36) are rejected by some S3-compatible stores.
        config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required"),
    )


def bucket() -> str:
    return os.environ.get("S3_BUCKET", "fineretouch")


def get_bytes(key: str) -> bytes:
    return _client().get_object(Bucket=bucket(), Key=key)["Body"].read()


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    _client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)
