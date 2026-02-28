PARAKEET_MODEL ?= nemo-parakeet-tdt-0.6b-v2
VENV_PKGS       = .venv/lib/python3.13/site-packages
TRT_LIBS        = $(VENV_PKGS)/tensorrt_libs
TRT_INCLUDE     = third_party/tensorrt
CUDA_HOME      ?= /usr/local/cuda-13.1
export LD_LIBRARY_PATH := $(TRT_LIBS):$(VENV_PKGS)/onnxruntime/capi:$(LD_LIBRARY_PATH)

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -march=native -flto -Wno-deprecated-declarations -I$(TRT_INCLUDE) -I$(CUDA_HOME)/include
LDFLAGS  = -flto -L$(CUDA_HOME)/lib64 -lcudart -lcufft $(TRT_LIBS)/libnvinfer.so.10 -Wl,-rpath,$(TRT_LIBS)

.PHONY: run bench bench-cpp engines

run: src/parakeet-transcribe.py
	arecord -f S16_LE -r 16000 -c 1 -t raw - | \
	  ORT_LOG_LEVEL=3 uv run src/parakeet-transcribe.py --model $(PARAKEET_MODEL) --cuda

engines: engines/encoder.engine engines/decoder_joint.engine

engines/encoder.engine engines/decoder_joint.engine: scripts/build_engines.py
	uv run python scripts/build_engines.py

bench: parakeet engines/encoder.engine engines/decoder_joint.engine
	@echo "=== Python (onnx-asr + TRT EP) ==="
	uv run python tests/bench.py
	@echo ""
	@echo "=== C++ (parakeet.cpp + TRT) ==="
	uv run python tests/bench_cpp.py

bench-cpp: parakeet engines/encoder.engine engines/decoder_joint.engine
	uv run python tests/bench_cpp.py

# C++ build
parakeet: src/parakeet.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
