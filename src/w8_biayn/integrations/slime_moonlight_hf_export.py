from __future__ import annotations

import argparse
import re
import runpy
import sys


MOONLIGHT_KV_LAYERNORM_PATTERN = re.compile(
    r"module\.module\.decoder\.layers\.(\d+)\.self_attention\.kv_layernorm\.weight"
)


def moonlight_deepseekv3_alias(name: str) -> str | None:
    match = MOONLIGHT_KV_LAYERNORM_PATTERN.fullmatch(name)
    if not match:
        return None
    layer_idx = match.group(1)
    return f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"


def _patch_slime_deepseekv3_converter() -> None:
    import slime.backends.megatron_utils.megatron_to_hf as megatron_to_hf
    import slime.backends.megatron_utils.megatron_to_hf.deepseekv3 as deepseekv3

    original = megatron_to_hf.convert_deepseekv3_to_hf

    def patched(args, name, param):  # type: ignore[no-untyped-def]
        alias = moonlight_deepseekv3_alias(name)
        if alias is not None:
            return [(alias, param)]
        return original(args, name, param)

    megatron_to_hf.convert_deepseekv3_to_hf = patched
    deepseekv3.convert_deepseekv3_to_hf = patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SLIME torch-dist to HF export with Moonlight converter aliases.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--slime-root", default="/root/slime")
    parser.add_argument("--model-name")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--add-missing-from-origin-hf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _patch_slime_deepseekv3_converter()

    converter_argv = [
        f"{args.slime_root}/tools/convert_torch_dist_to_hf.py",
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--origin-hf-dir",
        args.origin_hf_dir,
    ]
    if args.model_name:
        converter_argv.extend(["--model-name", args.model_name])
    if args.chunk_size is not None:
        converter_argv.extend(["--chunk-size", str(args.chunk_size)])
    if args.vocab_size is not None:
        converter_argv.extend(["--vocab-size", str(args.vocab_size)])
    if args.force:
        converter_argv.append("--force")
    if args.add_missing_from_origin_hf:
        converter_argv.append("--add-missing-from-origin-hf")

    sys.argv = converter_argv
    runpy.run_path(converter_argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
