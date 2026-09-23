"""Modal app (CPU): copy ONE directory of the `maemm-data` volume to the owner's S3 bucket and return presigned URLs.
Generic counterpart of data/modal_export_s3_v2.py (which exports the fixed research-data bundle). Credentials are CALL
ARGUMENTS, never stored in code, in the image or on the volume.

    cd ~/maemm-pub-simple2m && MODAL_PROFILE=safety-sahan modal deploy data/modal_export_dir_s3.py
    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 scripts/launchers/spawn_export_dir_s3.py \
        /data/sft_mix/simple2m_sft/final sft-init-simple2m
"""
import os

import modal

app = modal.App(os.environ.get("DIR_S3_APP", "maemm-export-dir-s3"))
image = modal.Image.debian_slim(python_version="3.12").pip_install("boto3")
vol = modal.Volume.from_name("maemm-data")


@app.function(image=image, cpu=16, memory=32 * 1024, volumes={"/data": vol}, timeout=8 * 3600)
def export_dir(src_dir: str, bucket: str, prefix: str, aws_key: str, aws_secret: str, region: str = "us-east-1",
               presign_days: int = 7, skip: tuple = ()):
    """Upload every file of `src_dir` (one level, no recursion into subdirs) to s3://<bucket>/<prefix>/, write a
    urls.txt next to them, and return every presigned URL."""
    import time
    import boto3
    from boto3.s3.transfer import TransferConfig
    t0 = time.time(); vol.reload()
    log = lambda m: print(f"[dir-s3 +{time.time() - t0:5.0f}s] {m}", flush=True)
    assert os.path.isdir(src_dir), src_dir
    files = sorted(f for f in os.listdir(src_dir) if os.path.isfile(f"{src_dir}/{f}") and f not in skip)
    total = sum(os.path.getsize(f"{src_dir}/{f}") for f in files)
    log(f"{src_dir} -> s3://{bucket}/{prefix}/: {len(files)} files, {total / 2**30:.1f} GiB")
    s3 = boto3.client("s3", region_name=region, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:  # noqa
        s3.create_bucket(Bucket=bucket) if region == "us-east-1" else \
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        log(f"created bucket {bucket}")
    cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16)
    urls, done = [], 0
    for f in files:
        key = f"{prefix}/{f}"
        s3.upload_file(f"{src_dir}/{f}", bucket, key, Config=cfg)
        size = os.path.getsize(f"{src_dir}/{f}"); done += size
        urls.append((key, size, s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=presign_days * 86400)))
        if size > 2**30:
            log(f"uploaded {f} ({size / 2**30:.1f} GiB, {done / 2**30:.1f}/{total / 2**30:.1f})")
    body = (f"# {src_dir} -> s3://{bucket}/{prefix}/  ({len(urls)} files, {total / 2**30:.1f} GiB); presigned {presign_days} days\n"
            + "\n".join(f"{k}\t{b}\t{u}" for k, b, u in urls) + "\n")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/urls.txt", Body=body.encode())
    log(f"DONE: {len(urls)} files, {total / 2**30:.1f} GiB")
    return {"src_dir": src_dir, "bucket": bucket, "prefix": prefix, "n_files": len(urls), "gib": round(total / 2**30, 2),
            "wall_s": round(time.time() - t0), "urls": urls}
