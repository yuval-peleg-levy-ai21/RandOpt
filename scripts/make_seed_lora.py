#!/usr/bin/env python3
"""
Generate *random seed LoRA adapters* for a model and wire them into an existing
algo-model-serving variant YAML.

This is the LoRA-native equivalent of RandOpt's seed-based weight perturbation
(see ``utils/worker_extn.py::perturb_self_weights``). Instead of regenerating a
full-shape noise tensor for every parameter on-GPU via ``collective_rpc``, we
materialise each perturbation ahead of time as a PEFT LoRA adapter that vLLM can
serve via its native ``--enable-lora`` API (selected per request through the
OpenAI ``model`` field, e.g. ``model=seed7``).

For every targeted linear layer (weight shape ``[out_features, in_features]``)
we draw a *single* random vector of length ``rank * (in_features + out_features)``
from ``torch.randn`` seeded with ``int(seed)`` -- exactly as
``es_worker_extension.WorkerExtension.perturb_self_weights`` seeds every parameter
(``gen.manual_seed(int(seed))``, the same seed for all modules) -- then split it
into the two LoRA factors:

    A = first  rank * in_features  values  -> reshape (rank, in_features)
    B = last   rank * out_features values  -> reshape (out_features, rank)

Each factor is scaled by ``sqrt(noise_scale * sqrt(rank) / alpha)`` so the
PEFT-applied delta ``(alpha / rank) * (B @ A)`` has per-element std ``noise_scale``
-- the same magnitude as the ES perturbation ``noise_scale * N(0,1)``. The sqrt is
because the delta is a *product* of two random factors, so each must carry the
square root of the target scale (for the default rank=1, alpha=rank this is simply
``sqrt(noise_scale)``). The same ``(model, seed, rank, alpha, noise_scale)`` always
reproduces the same adapter.

Note: mirroring the verl perturb, every module uses the same seed, so targeted
layers with identical (in+out) dimensions receive identical factors.

Given a path to an existing variant YAML and a seed count, this script:
  1. Reads the model identity (HF_MODEL_ID / weights.path) from the YAML.
  2. Generates ``num_seeds`` adapters (seed0 .. seedN-1) under ``workdir``.
  3. Copies them to a ``<weights.path stem>-seed-loras/`` sibling dir in GCS
     (skip with ``--no_upload``).
  4. Writes a copy of the YAML with ENABLE_LORA / LORA_MODULES /
     additionalArtifacts added so it serves the adapters.

Example:
    python3 scripts/make_seed_lora.py \\
        --variant_yaml ../algo-model-serving/helm/values/variant/qwen.yaml \\
        --num_seeds 20
"""
import fire
import json
import math
import os
import subprocess
from typing import Optional

import torch

# Default LoRA targets: the attention + MLP projections every transformer
# exposes as separate nn.Linear layers. Override with --target_modules.
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Where the per-model adapters live and how they are referenced once fetched.
# additionalArtifacts copies gs://<dir>/* into /artifacts/<KEY>/, so a single
# key covers all seeds: /artifacts/SEED_LORAS/seedN.
_ARTIFACT_KEY = "SEED_LORAS"
_ADAPTER_PREFIX = "seed"
_SEED_LORA_SUFFIX = "-seed-loras"


# ==================== Variant YAML I/O ====================


def _make_yaml():
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # don't auto-wrap; yamllint allows up to 400 and we fold manually
    return yaml


def _load_variant(yaml, path):
    with open(path) as variant_file:
        data = yaml.load(variant_file)
    return data


def _models_node(data):
    """Return global.defaults.models, erroring clearly if the layout is unexpected."""
    try:
        return data["global"]["defaults"]["models"]
    except (KeyError, TypeError) as error:
        raise SystemExit(
            f"Unexpected variant layout: missing global.defaults.models ({error})."
        )


def _read_env_var(data, models, name):
    """Read an env var from the model-level envVars, falling back to global-level."""
    model_env = models.get("deployment", {}).get("envVars", {})
    if name in model_env:
        return str(model_env[name])
    global_env = data.get("global", {}).get("deployment", {}).get("envVars", {})
    if name in global_env:
        return str(global_env[name])
    raise SystemExit(f"Variant YAML does not define {name}.")


def _seed_lora_gcs_path(weights_path):
    """Sibling of the model's weights dir, e.g. '.../qwen25-32b-seed-loras/'.

    A sibling (not a child) keeps the adapters out of the model-fetch glob
    'gs://<model dir>/*', which would otherwise download them into /model too.
    """
    return weights_path.rstrip("/") + _SEED_LORA_SUFFIX + "/"


def _lora_modules_value(adapter_names):
    """Single-line space-separated 'seedN=/artifacts/<KEY>/seedN', as vLLM and
    register_model.sh consume it directly. Note: with many seeds this line can
    exceed yamllint's max-line-length (raise the limit / disable the rule for it).
    """
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    value = " ".join(f"{name}=/artifacts/{_ARTIFACT_KEY}/{name}" for name in adapter_names)
    return DoubleQuotedScalarString(value)


def _update_variant(data, models, adapter_names, gcs_artifact_path):
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    weights = models["weights"]
    if not weights.get("additionalArtifacts"):
        weights["additionalArtifacts"] = CommentedMap()
    weights["additionalArtifacts"][_ARTIFACT_KEY] = gcs_artifact_path

    # LORA_MODULES must reach two containers from two separate ConfigMaps: the
    # per-model block feeds the vLLM container (--lora-modules), the top-level
    # global block feeds the registration container (register_model.sh, which
    # registers each adapter as a gateway endpoint). Anchor it once and alias it,
    # matching the deepseek-ai-deepseek-ocr variant. ENABLE_LORA only needs the
    # per-model block (the vLLM command), not registration.
    lora_modules = _lora_modules_value(adapter_names)
    lora_modules.yaml_set_anchor("LORA_MODULES", always_dump=True)

    global_env = data["global"].setdefault("deployment", CommentedMap()).setdefault("envVars", CommentedMap())
    global_env["LORA_MODULES"] = lora_modules
    global_env["TOOL_SUPPORT"] = "true"
    # The single-line value exceeds yamllint's max-line-length with many seeds.
    global_env.yaml_add_eol_comment("# yamllint disable-line rule:line-length", "LORA_MODULES")

    model_env = models["deployment"]["envVars"]
    model_env["ENABLE_LORA"] = DoubleQuotedScalarString("true")
    model_env["LORA_MODULES"] = lora_modules
    model_env["TOOL_CALL_PARSER"] = "hermes"
    model_env["ENABLE_AUTO_TOOL_CHOICE"] = "true"



def _write_variant(yaml, data, path):
    with open(path, "w") as out_file:
        yaml.dump(data, out_file)


# ==================== Adapter generation ====================


def _build_meta_model(config_source, trust_remote_code):
    """Instantiate the model skeleton on the meta device (no weights, no GPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(config_source, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return config, model


def _collect_target_linears(model, target_modules):
    """Map each targeted linear module's path to its (out_features, in_features)."""
    targets = set(target_modules)
    linears = {}
    for name, module in model.named_modules():
        if name.split(".")[-1] not in targets:
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            continue
        linears[name] = (out_features, in_features)
    return linears


def _generate_lora_pair(out_features, in_features, rank, seed, factor, dtype):
    """Draw one combined vector for this layer and split it into LoRA A and B.

    Seeded with ``int(seed)`` exactly like perturb_self_weights
    (``gen.manual_seed(int(seed))``) -- the same seed for every module. Each factor
    is scaled by ``factor`` so the PEFT delta ``(alpha/rank)*(B@A)`` has per-element
    std equal to the target ``noise_scale``.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    combined = torch.randn(rank * (in_features + out_features), generator=generator)
    combined.mul_(factor)

    split_at = rank * in_features
    lora_a = combined[:split_at].reshape(rank, in_features).to(dtype)
    lora_b = combined[split_at:].reshape(out_features, rank).to(dtype)
    return lora_a, lora_b


def _build_adapter_tensors(linears, base_seed, rank, factor, dtype):
    """Build the PEFT-format state dict for all targeted layers for one seed."""
    tensors = {}
    for module_path, (out_features, in_features) in linears.items():
        lora_a, lora_b = _generate_lora_pair(
            out_features, in_features, rank, base_seed, factor, dtype
        )
        key_prefix = f"base_model.model.{module_path}"
        tensors[f"{key_prefix}.lora_A.weight"] = lora_a
        tensors[f"{key_prefix}.lora_B.weight"] = lora_b
    return tensors


def _build_adapter_config(model_name, target_modules, rank, alpha):
    """Build a PEFT LoraConfig serialisation accepted by PEFT and vLLM."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "auto_mapping": None,
        "base_model_name_or_path": model_name,
        "revision": None,
        "inference_mode": True,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
    }


def _resolve_dtype(dtype_arg, config):
    if dtype_arg is not None:
        return _DTYPES[dtype_arg]
    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    return torch.bfloat16


def _resolve_config_source(model_path, hf_model_id):
    """Use the local model dir if it exists, else the HF hub id from the YAML."""
    if model_path and os.path.isdir(model_path):
        return model_path
    return hf_model_id


def _write_adapter(adapter_dir, tensors, adapter_config):
    from safetensors.torch import save_file

    os.makedirs(adapter_dir, exist_ok=True)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as config_file:
        json.dump(adapter_config, config_file, indent=2)


def _generate_all_adapters(
    config_source, model_name, target_modules, num_seeds, workdir, rank, alpha, noise_scale, dtype_arg, trust_remote_code
):
    """Build the meta model once, then write one adapter per seed. Returns names."""
    config, model = _build_meta_model(config_source, trust_remote_code)
    dtype = _resolve_dtype(dtype_arg, config)

    linears = _collect_target_linears(model, target_modules)
    if not linears:
        raise SystemExit(
            f"No linear layers matched target modules {list(target_modules)}. "
            "Inspect the model and pass --target_modules explicitly."
        )

    # Per-factor std so the PEFT delta (alpha/rank)*(B@A) has per-element std ==
    # noise_scale, matching perturb_self_weights' noise_scale * N(0,1). The sqrt is
    # because the delta is a product of two random factors:
    #   Var[(B@A)_ij] = rank * f**4  ->  std[delta] = (alpha/sqrt(rank)) * f**2.
    factor = math.sqrt(noise_scale * math.sqrt(rank) / alpha)

    adapter_config = _build_adapter_config(model_name, target_modules, rank, alpha)
    adapter_names = []
    for seed in range(num_seeds):
        name = f"{_ADAPTER_PREFIX}{seed}"
        tensors = _build_adapter_tensors(linears, seed, rank, factor, dtype)
        _write_adapter(os.path.join(workdir, name), tensors, adapter_config)
        adapter_names.append(name)

    print(
        f"Generated {len(adapter_names)} adapters in {workdir}\n"
        f"  rank={rank}  alpha={alpha}  noise_scale={noise_scale:.4g}  "
        f"factor(std of A,B)={factor:.4g}  (PEFT scale alpha/rank={alpha / rank:.4g})\n"
        f"  effective per-weight delta std ~= {noise_scale:.4g}\n"
        f"  dtype={str(dtype).replace('torch.', '')}  layers={len(linears)}"
    )
    return adapter_names


def _upload_to_gcs(workdir, adapter_names, gcs_artifact_path):
    """Copy each generated adapter dir to gs://<artifact path>/seedN/."""
    destination = "gs://" + gcs_artifact_path
    sources = [os.path.join(workdir, name) for name in adapter_names]
    print(f"Uploading {len(sources)} adapters to {destination}")
    subprocess.run(["gsutil", "-m", "cp", "-r", *sources, destination], check=True)


def _default_output_yaml(input_path):
    root, ext = os.path.splitext(input_path)
    return f"{root}-seedlora{ext}"


def make_lora_seeds(
    variant_yaml: str,
    num_seeds: int,
    workdir: str = "./seed_loras_build",
    output_yaml: Optional[str] = None,
    model_path: Optional[str] = None,
    rank: int = 1,
    alpha: Optional[float] = None,
    noise_scale: float = 1.0,
    target_modules=_DEFAULT_TARGET_MODULES,
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    no_upload: bool = False,
):
    """Generate seed LoRA adapters and wire them into a copy of the variant YAML.

    Args:
        variant_yaml: Path to an existing algo-model-serving variant YAML.
        num_seeds: Number of adapters to generate (seeds 0 .. N-1).
        workdir: Local directory to write the generated adapters into.
        output_yaml: Where to write the updated YAML. Defaults to <input>-seedlora.yaml.
        model_path: Local model dir to read config.json from; falls back to HF_MODEL_ID.
        rank: LoRA rank.
        alpha: LoRA alpha (PEFT scaling alpha/rank). The factor magnitude compensates
            for it, so it cancels out of the effective delta std. Defaults to rank.
        noise_scale: Target per-element std of the weight delta, matching
            es_worker_extension.perturb_self_weights's noise_scale. Each LoRA factor
            is drawn at std sqrt(noise_scale * sqrt(rank) / alpha) (= sqrt(noise_scale)
            for the default rank=1, alpha=rank), since the delta is a product B@A.
        target_modules: Linear module leaf names to perturb.
        dtype: Adapter tensor dtype (bfloat16/float16/float32). Defaults to the model's.
        trust_remote_code: Allow loading custom modeling code for non-standard architectures.
        no_upload: Skip the gsutil upload; only generate adapters and update the YAML.
    """
    if rank < 1:
        raise SystemExit("--rank must be >= 1")
    if num_seeds < 1:
        raise SystemExit("--num_seeds must be >= 1")
    if noise_scale <= 0:
        raise SystemExit("--noise_scale must be > 0")
    if dtype is not None and dtype not in _DTYPES:
        raise SystemExit(f"--dtype must be one of {sorted(_DTYPES)}")
    alpha = float(alpha) if alpha is not None else float(rank)

    yaml = _make_yaml()
    data = _load_variant(yaml, variant_yaml)
    models = _models_node(data)

    hf_model_id = _read_env_var(data, models, "HF_MODEL_ID")
    weights_path = str(models["weights"]["path"])
    gcs_artifact_path = _seed_lora_gcs_path(weights_path)

    config_source = _resolve_config_source(model_path, hf_model_id)
    adapter_names = _generate_all_adapters(
        config_source, hf_model_id, list(target_modules), num_seeds, workdir, rank, alpha, noise_scale, dtype, trust_remote_code
    )

    if no_upload:
        print(f"Skipped upload (--no_upload). Intended GCS location: gs://{gcs_artifact_path}")
    else:
        _upload_to_gcs(workdir, adapter_names, gcs_artifact_path)

    _update_variant(data, models, adapter_names, gcs_artifact_path)
    output_yaml = output_yaml or _default_output_yaml(variant_yaml)
    _write_variant(yaml, data, output_yaml)

    print(
        f"Wrote updated variant to {output_yaml}\n"
        f"  additionalArtifacts.{_ARTIFACT_KEY}={gcs_artifact_path}\n"
        f"  LORA_MODULES references /artifacts/{_ARTIFACT_KEY}/{_ADAPTER_PREFIX}0.."
        f"{_ADAPTER_PREFIX}{num_seeds - 1}\n"
        "Next: run the repo's `inv sort-env-vars --fix` + `pytest` to normalise the variant."
    )


if __name__ == "__main__":
    fire.Fire(make_lora_seeds)