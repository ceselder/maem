"""Multi-node deployment entry for sft/modal_sft.py -- pins the app name so NOTHING launched through this file can
resolve to the production single-node app (maemm-sft-8xb200). Always deploy/run multi-node work via this file:

    export SFT_TRANSFORMERS="transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3"  # --prefix-cache
    SFT_N_NODES=4 modal deploy sft/modal_sft_multinode.py            # 4 x B200:8 cluster (SFT_MN_GPU / SFT_MN_RDMA / SFT_DISK_GB also deploy-time)
    modal run sft/modal_sft_multinode.py::run_cluster_check           # cheap: cluster info + 32-rank NCCL all-reduce + volume read speed
    modal run sft/modal_sft_multinode.py::launch_multinode --run-name ... --data-dir ... --extra-args "... --grad-accum 8"

Defaults set here (override by exporting before the command):
    SFT_APP_NAME=maemm-sft-multinode      the ONLY app this file deploys to
    SFT_SUPERVISOR=0                      the single-node auto-resume supervisor stays with the production deployment
    SFT_MULTINODE_SUPERVISOR=1            the multi-node auto-resume supervisor runs here
"""
import os
import sys

os.environ.setdefault("SFT_APP_NAME", "maemm-sft-multinode")
os.environ.setdefault("SFT_SUPERVISOR", "0")
os.environ.setdefault("SFT_MULTINODE_SUPERVISOR", "1")
assert os.environ["SFT_APP_NAME"] != "maemm-sft-8xb200", \
    "refusing to deploy multi-node code onto the production single-node app"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_sft import *  # noqa: E402,F401,F403  -- re-exports every function / local entrypoint
from modal_sft import app  # noqa: E402,F401  -- explicit: `modal deploy` looks this name up
