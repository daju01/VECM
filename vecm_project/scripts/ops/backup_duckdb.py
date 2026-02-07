from __future__ import annotations

import argparse
import datetime as dt
import importlib
import shutil
from pathlib import Path

from vecm_project.scripts import storage


def _backup_local(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"vecm_duckdb_{timestamp}.duckdb"
    shutil.copy2(storage.DB_PATH, backup_path)
    return backup_path


def _upload_s3(path: Path, bucket: str, key_prefix: str) -> None:
    if importlib.util.find_spec("boto3") is None:
        raise RuntimeError("boto3 not installed; cannot upload to S3")
    boto3 = importlib.import_module("boto3")
    client = boto3.client("s3")
    key = f"{key_prefix}/{path.name}".lstrip("/")
    client.upload_file(str(path), bucket, key)


def _upload_gcs(path: Path, bucket: str, key_prefix: str) -> None:
    if importlib.util.find_spec("google.cloud.storage") is None:
        raise RuntimeError("google-cloud-storage not installed; cannot upload to GCS")
    gcs_storage = importlib.import_module("google.cloud.storage")
    client = gcs_storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(f"{key_prefix}/{path.name}".lstrip("/"))
    blob.upload_from_filename(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup DuckDB database to local/S3/GCS.")
    parser.add_argument("--output-dir", default="backups", help="Local output directory")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket name")
    parser.add_argument("--gcs-bucket", default=None, help="GCS bucket name")
    parser.add_argument("--key-prefix", default="vecm", help="Remote key prefix")
    args = parser.parse_args()

    backup_path = _backup_local(Path(args.output_dir))
    if args.s3_bucket:
        _upload_s3(backup_path, args.s3_bucket, args.key_prefix)
    if args.gcs_bucket:
        _upload_gcs(backup_path, args.gcs_bucket, args.key_prefix)


if __name__ == "__main__":
    main()
