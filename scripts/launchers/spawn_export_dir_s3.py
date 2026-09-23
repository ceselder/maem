"""Spawn data/modal_export_dir_s3.export_dir for one volume directory. Reads the AWS default profile from
~/.aws/credentials at call time and passes the keys as call arguments; nothing is printed or stored except the call id.

    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 scripts/launchers/spawn_export_dir_s3.py <src_dir> <prefix>
"""
import configparser
import json
import os
import sys
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
src_dir, prefix = sys.argv[1], sys.argv[2]
cp = configparser.ConfigParser(); cp.read(os.path.expanduser("~/.aws/credentials"))
key, secret = cp["default"]["aws_access_key_id"], cp["default"]["aws_secret_access_key"]
bucket = os.environ.get("ARB_BUCKET", "celeste-maemm-27b-data")
fc = modal.Function.from_name(os.environ.get("DIR_S3_APP", "maemm-export-dir-s3"), "export_dir").spawn(
    src_dir=src_dir, bucket=bucket, prefix=prefix, aws_key=key, aws_secret=secret, region="us-east-1", presign_days=7)
rec = {"call": fc.object_id, "src_dir": src_dir, "bucket": bucket, "prefix": prefix, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
p = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("s3_dir_exports", {})[prefix] = rec; json.dump(d, open(p, "w"), indent=1)
print(json.dumps(rec))
