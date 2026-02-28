#!/usr/bin/env python3
"""Build TensorRT engines from ONNX models for parakeet-tdt-0.6b-v2."""

import os
from pathlib import Path

import tensorrt as trt

REPO_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx"
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engines"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_onnx_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_ID))


def build_engine(onnx_path: Path, profiles: list[dict], engine_path: Path) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing {onnx_path.name} ...")
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 5

    for profile_spec in profiles:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in profile_spec.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    print(f"Building {engine_path.name} (this may take a few minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {engine_path.name}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"Wrote {engine_path} ({engine_path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    onnx_dir = get_onnx_dir()

    # Encoder: dynamic audio length
    build_engine(
        onnx_dir / "encoder-model.onnx",
        profiles=[
            {
                "audio_signal": ((1, 128, 5), (1, 128, 2000), (1, 128, 12000)),
                "length": ((1,), (1,), (1,)),
            }
        ],
        engine_path=ENGINE_DIR / "encoder.engine",
    )

    # Decoder+Joint: all fixed shapes
    build_engine(
        onnx_dir / "decoder_joint-model.onnx",
        profiles=[
            {
                "encoder_outputs": ((1, 1024, 1), (1, 1024, 1), (1, 1024, 1)),
                "targets": ((1, 1), (1, 1), (1, 1)),
                "target_length": ((1,), (1,), (1,)),
                "input_states_1": ((2, 1, 640), (2, 1, 640), (2, 1, 640)),
                "input_states_2": ((2, 1, 640), (2, 1, 640), (2, 1, 640)),
            }
        ],
        engine_path=ENGINE_DIR / "decoder_joint.engine",
    )

    print("Done.")


if __name__ == "__main__":
    main()
