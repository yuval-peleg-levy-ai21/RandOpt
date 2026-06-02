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
seeded deterministically from ``(seed, module_name)``, then split it into the
two LoRA factors:

    A = first  rank * in_features  values  -> reshape (rank, in_features)
    B = last   rank * out_features values  -> reshape (out_features, rank)

The effective weight delta applied at inference is the PEFT-standard
``(lora_alpha / rank) * (B @ A)``. The same ``(model, seed, rank, alpha)`` always
reproduces the same adapter.

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

import hashlib
import json
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
    """Folded block scalar: 'seedN=/artifacts/<KEY>/seedN', one per line.

    A folded scalar joins single newlines with spaces, so vLLM receives the
    space-separated form it expects while every line stays well under yamllint's
    400-char limit.
    """
    from ruamel.yaml.scalarstring import FoldedScalarString

    lines = "\n".join(f"{name}=/artifacts/{_ARTIFACT_KEY}/{name}" for name in adapter_names)
    return FoldedScalarString(lines)


def _update_variant(models, adapter_names, gcs_artifact_path):
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    weights = models["weights"]
    if not weights.get("additionalArtifacts"):
        weights["additionalArtifacts"] = CommentedMap()
    weights["additionalArtifacts"][_ARTIFACT_KEY] = gcs_artifact_path

    env_vars = models["deployment"]["envVars"]
    env_vars["ENABLE_LORA"] = DoubleQuotedScalarString("true")
    env_vars["LORA_MODULES"] = _lora_modules_value(adapter_names)


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


def _module_seed(base_seed, module_path):
    """Derive a stable, distinct seed per module so layers get independent noise."""
    digest = hashlib.sha256(f"{base_seed}:{module_path}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _generate_lora_pair(out_features, in_features, rank, module_seed, dtype):
    """Draw one combined vector for this layer and split it into LoRA A and B."""
    generator = torch.Generator(device="cpu").manual_seed(module_seed)
    combined = torch.randn(rank * (in_features + out_features), generator=generator)

    split_at = rank * in_features
    lora_a = combined[:split_at].reshape(rank, in_features).to(dtype)
    lora_b = combined[split_at:].reshape(out_features, rank).to(dtype)
    return lora_a, lora_b


def _build_adapter_tensors(linears, base_seed, rank, dtype):
    """Build the PEFT-format state dict for all targeted layers for one seed."""
    tensors = {}
    for module_path, (out_features, in_features) in linears.items():
        lora_a, lora_b = _generate_lora_pair(
            out_features, in_features, rank, _module_seed(base_seed, module_path), dtype
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
    config_source, model_name, target_modules, num_seeds, workdir, rank, alpha, dtype_arg, trust_remote_code
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

    adapter_config = _build_adapter_config(model_name, target_modules, rank, alpha)
    adapter_names = []
    for seed in range(num_seeds):
        name = f"{_ADAPTER_PREFIX}{seed}"
        tensors = _build_adapter_tensors(linears, seed, rank, dtype)
        _write_adapter(os.path.join(workdir, name), tensors, adapter_config)
        adapter_names.append(name)

    print(
        f"Generated {len(adapter_names)} adapters in {workdir}\n"
        f"  rank={rank}  alpha={alpha}  scale={alpha / rank:.4g}\n"
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


def main(
    variant_yaml: str,
    num_seeds: int,
    workdir: str = "./seed_loras_build",
    output_yaml: Optional[str] = None,
    model_path: Optional[str] = None,
    rank: int = 1,
    alpha: Optional[float] = None,
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
        alpha: LoRA alpha (effective scale = alpha/rank). Defaults to rank (scale 1.0).
        target_modules: Linear module leaf names to perturb.
        dtype: Adapter tensor dtype (bfloat16/float16/float32). Defaults to the model's.
        trust_remote_code: Allow loading custom modeling code for non-standard architectures.
        no_upload: Skip the gsutil upload; only generate adapters and update the YAML.
    """
    if rank < 1:
        raise SystemExit("--rank must be >= 1")
    if num_seeds < 1:
        raise SystemExit("--num_seeds must be >= 1")
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
        config_source, hf_model_id, list(target_modules), num_seeds, workdir, rank, alpha, dtype, trust_remote_code
    )

    if no_upload:
        print(f"Skipped upload (--no_upload). Intended GCS location: gs://{gcs_artifact_path}")
    else:
        _upload_to_gcs(workdir, adapter_names, gcs_artifact_path)

    _update_variant(models, adapter_names, gcs_artifact_path)
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
    import fire

    fire.Fire(main)