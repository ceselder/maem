"""Central config. Scales from K=10k (pilot) to K=1,000,000 clusters by changing CLUSTERS."""
import os
from dataclasses import dataclass, field

# Model profile: env MAEMM_MODEL_PROFILE, else the one-word file mxf/PROFILE (ships inside every Modal image that mounts mxf/),
# else qwen36-27b. Everything model-specific the pipeline branches on lives here.
_PROFILES = {
    "qwen36-27b": dict(MODEL="Qwen/Qwen3.6-27B", D_MODEL=5120, READ_LAYER=42, N_LAYERS=64, MULTIMODAL=True,
                       WHITEN_MU="/data/acts27b/whiten_mu.npy"),
    # Qwen3-8B + Adam Karvonen's BatchTopK SAE (adamkarvonen/qwen3-8b-saes, resid_post_layer_27 trainer_2: 65,536 features, k=80)
    "qwen3-8b": dict(MODEL="Qwen/Qwen3-8B", D_MODEL=4096, READ_LAYER=27, N_LAYERS=36, MULTIMODAL=False,
                     WHITEN_MU="/data/qwen3_8b/whiten_mu_l27.npy"),
}


def _profile_name():
    p = os.environ.get("MAEMM_MODEL_PROFILE", "").strip()
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROFILE")
    if not p and os.path.exists(f):
        p = open(f).read().strip()
    return p or "qwen36-27b"


PROFILE = _profile_name()
_P = _PROFILES[PROFILE]
MODEL = _P["MODEL"]
D_MODEL = _P["D_MODEL"]
READ_LAYER = _P["READ_LAYER"]  # layer the direction is read/maximized at (matches the SAE)
N_LAYERS = _P["N_LAYERS"]
MULTIMODAL = _P["MULTIMODAL"]  # Qwen3.6 = Qwen3_5ForConditionalGeneration (text weights under model.language_model.*, vision shard)
LM_PREFIX = "model.language_model." if MULTIMODAL else "model."   # checkpoint key prefix of the decoder stack
WHITEN_MU = _P["WHITEN_MU"]    # layer-READ_LAYER corpus mean: realact direction centering + the centered scorer
INJECT_LAYER = 1       # activation-oracle injection site
STEER_COEFF = 1.0      # norm-matched addition scale
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # fast clusterer; swap to bge-small for max speed
CORPUS = "openbmb/Ultra-FineWeb"         # HF streaming; en subset


@dataclass
class ClusterConfig:
    corpus: str = CORPUS
    embed_model: str = EMBED_MODEL
    n_docs: int = 4_000_000       # docs to embed (rule of thumb: >=40*K for stable k-means)
    clusters: int = 10_000        # PILOT. Scale target: 1_000_000.
    min_chars: int = 200          # skip tiny docs
    max_chars: int = 2000         # truncate for embedding
    embed_batch: int = 1024
    shard_size: int = 500_000     # embeddings per memmap shard on disk
    out_dir: str = "data/clusters"
    seed: int = 0


@dataclass
class ProbeCacheConfig:
    members_per_cluster: int = 64     # texts per cluster pushed through Qwen3 for probe fitting
    pool: str = "mean"                # "mean" | "last" residual pooling at READ_LAYER
    batch: int = 128
    out_dir: str = "data/resid_cache"


@dataclass
class BuildDataConfig:
    n_examples: int = 8_000_000       # (direction, target_text) training pairs to mint
    targets_per_example: int = 5      # centroid-closest texts used as SFT targets
    probe_c: float = 1.0              # logistic-regression inverse-reg
    negatives_per_probe: int = 512    # cluster-B members used as the probe's negative class
    pair_mode: str = "AvsB"           # "AvsB" (cluster vs cluster) | "AvsRest"
    shard_examples: int = 250_000
    out_dir: str = "data/pretrain"
    seed: int = 0


@dataclass
class TrainConfig:
    init_adapter: str | None = None   # None = fresh rsLoRA on base; else continue
    lora_r: int = 64
    lora_alpha: int = 16              # rsLoRA: 16/sqrt(64)=2
    lora_dropout: float = 0.0         # RL requires 0; keep 0 for pretrain too (simplicity)
    lr: float = 3e-5
    batch_size: int = 64
    epochs: int = 1
    max_seq: int = 192
    warmup_frac: float = 0.02
    save_dir: str = "checkpoints/pretrain"
    run_name: str = "mxf-pretrain"


@dataclass
class RLConfig:
    """Dr. GRPO — no /std, no KL, global-token normalizer."""
    init_adapter: str = "checkpoints/pretrain/final"
    groups_per_step: int = 256        # directions per step (global)
    group_size: int = 8
    lr: float = 1e-6
    clip_eps: float = 0.2
    tis_cap: float = 2.0              # TIS upper ratio cap — absorbs residual vLLM/HF kernel mismatch
    entropy_coef: float = 0.0         # β in maximize r + β·H(π); explicit diversity knob (no KL)
    max_new_tokens: int = 96
    min_new_tokens: int = 16
    temperature: float = 1.0
    total_steps: int = 30_000
    sync_every: int = 10              # push LoRA-merged actor weights into vLLM every N steps
    fluency_floor: float | None = -4.5   # optional gates (stability without KL)
    distinct_floor: float | None = 0.5
    gate_penalty: float = 25.0
    len_penalty_start: int | None = 64
    len_penalty_per_tok: float = 0.5
    direction_source: str = "cluster"    # "cluster" | "sae" | "mix"
    save_dir: str = "checkpoints/rl"
    run_name: str = "mxf-rl-drgrpo"


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    cache: ProbeCacheConfig = field(default_factory=ProbeCacheConfig)
    build: BuildDataConfig = field(default_factory=BuildDataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
