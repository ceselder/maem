"""Upload a FULL-MODEL checkpoint directory (HF-format safetensors shards + tokenizer/config) from the Modal volume `maemm-data`
to a HuggingFace model repo, from inside Modal (datacenter bandwidth, hf_transfer). The HF token is passed at call time, never stored.

    modal deploy scripts/modal_hf_upload_full.py            # app maemm-hf-upload-full
    python3 - <<'EOF'
    import modal
    fc = modal.Function.from_name("maemm-hf-upload-full", "upload_dir").spawn(
        src_dir="/data/sft_mix/<run>/final", repo="ceselder/<name>", hf_token=TOKEN, readme=open("README.md").read(),
        skip=("SAVE_DONE", "optim.pt"), private=False)
    EOF
Small extras (plots, json) are cheaper to push from the local box with HfApi.upload_file.
"""
import os

import modal

app = modal.App(os.environ.get("HF_UPLOAD_APP", "maemm-hf-upload-full"))
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub[hf_transfer]==0.34.4")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=image, cpu=8, memory=32768, volumes={"/data": vol}, timeout=8 * 3600)
def upload_dir(src_dir: str, repo: str, hf_token: str, readme: str = "", skip: tuple = ("SAVE_DONE", "optim.pt", "heartbeat"),
               private: bool = False, path_in_repo: str = "") -> dict:
    import time
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    vol.reload()
    assert os.path.isdir(src_dir), src_dir
    files = sorted(f for f in os.listdir(src_dir) if os.path.isfile(f"{src_dir}/{f}") and f not in skip)
    total = sum(os.path.getsize(f"{src_dir}/{f}") for f in files)
    print(f"[hf-upload-full] {src_dir} -> {repo}: {len(files)} files, {total / 2**30:.1f} GiB", flush=True)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    t0 = time.time()
    # upload_folder with allow_patterns streams the shards straight from the volume mount; one commit per call.
    api.upload_folder(folder_path=src_dir, repo_id=repo, path_in_repo=path_in_repo or ".", allow_patterns=files,
                      commit_message=f"full-model checkpoint from {src_dir}")
    if readme:
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, commit_message="model card")
    out = {"repo": repo, "files": files, "gib": round(total / 2**30, 2), "seconds": round(time.time() - t0, 1)}
    print(f"[hf-upload-full] done: {out}", flush=True)
    return out


@app.function(image=image, cpu=8, memory=32768, volumes={"/data": vol}, timeout=24 * 3600)
def upload_many(items: list, repo: str, hf_token: str, readme: str = "", private: bool = False,
                skip: tuple = ("SAVE_DONE", "optim.pt", "heartbeat")) -> dict:
    """Upload several checkpoint directories into ONE repo, sequentially (one commit each, so concurrent commits to the
    same ref never race). items = [[src_dir, path_in_repo], ...]; path_in_repo "." = repo root."""
    import time
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token); vol.reload()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    done, t0 = [], time.time()
    for src_dir, path_in_repo in items:
        assert os.path.isdir(src_dir), src_dir
        files = sorted(f for f in os.listdir(src_dir) if os.path.isfile(f"{src_dir}/{f}") and f not in skip)
        gib = sum(os.path.getsize(f"{src_dir}/{f}") for f in files) / 2**30
        t1 = time.time()
        api.upload_folder(folder_path=src_dir, repo_id=repo, path_in_repo=path_in_repo, allow_patterns=files,
                          commit_message=f"checkpoint {path_in_repo} from {src_dir}")
        done.append({"src": src_dir, "path": path_in_repo, "gib": round(gib, 2), "s": round(time.time() - t1, 1)})
        print(f"[upload-many] {path_in_repo}: {gib:.1f} GiB in {time.time() - t1:.0f}s ({len(done)}/{len(items)})", flush=True)
    if readme:
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, commit_message="model card")
    return {"repo": repo, "n": len(done), "gib": round(sum(d["gib"] for d in done), 1), "seconds": round(time.time() - t0), "items": done}
