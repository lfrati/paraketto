PARAKEET_MODEL ?= nemo-parakeet-tdt-0.6b-v2
VENV_PKGS       = .venv/lib/python3.13/site-packages
TRT_LIBS        = $(VENV_PKGS)/tensorrt_libs
TRT_INCLUDE     = third_party/tensorrt
CUDA_HOME      ?= /usr/local/cuda-13.1
export LD_LIBRARY_PATH := $(TRT_LIBS):$(VENV_PKGS)/onnxruntime/capi:$(LD_LIBRARY_PATH)

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(TRT_INCLUDE) -I$(CUDA_HOME)/include -Ithird_party -Isrc
NVFLAGS  = -std=c++17 -O3 -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcufft -lcublas -lpthread $(TRT_LIBS)/libnvinfer.so.10 -Wl,-rpath,$(TRT_LIBS)

.PHONY: bench-all bench-py bench-cpp bench-cuda engines inspect-onnx weights download-data clean

download-data:
	@if [ -d data/librispeech ]; then echo "data/ already exists"; else \
		gh release download bench-data --pattern 'bench-data.tar.gz' && \
		tar xzf bench-data.tar.gz && \
		rm bench-data.tar.gz && \
		echo "Downloaded $$(find data/ -name '*.wav' | wc -l) wav files"; \
	fi

engines: engines/encoder.engine engines/decoder_joint.engine

engines/encoder.engine engines/decoder_joint.engine: scripts/build_engines.py
	uv run python scripts/build_engines.py

BENCH_SEP = @printf '\n%s\n%s\n%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' '  $(1)'

bench-all: parakeet engines/encoder.engine engines/decoder_joint.engine parakeet.cuda
	$(call BENCH_SEP,Python  ·  ONNX Runtime + TRT EP)
	@uv run python tests/bench.py
	$(call BENCH_SEP,C++ TRT ·  parakeet.cpp + TensorRT)
	@uv run python tests/bench_cpp.py
	$(call BENCH_SEP,C++ CUDA · parakeet_cuda.cpp + cuBLAS)
	@uv run python tests/bench_cuda.py

bench-py:
	uv run python tests/bench.py

bench-cpp: parakeet engines/encoder.engine engines/decoder_joint.engine
	uv run python tests/bench_cpp.py

bench-cuda: parakeet.cuda
	uv run python tests/bench_cuda.py

# ONNX inspection and weight export
inspect-onnx:
	uv run python scripts/inspect_onnx.py

weights: weights.bin

weights.bin: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

# TRT backend (reference)
parakeet: src/parakeet.cpp src/kernels.o src/kernels.h
	$(CXX) $(CXXFLAGS) src/parakeet.cpp src/kernels.o $(LDFLAGS) -o $@

# CUDA backend (no TensorRT dependency)
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lpthread

parakeet.cuda: src/parakeet_cuda.cpp src/conformer.cpp src/conformer.h src/kernels.o src/kernels.h
	$(CXX) $(CUDA_CXXFLAGS) src/parakeet_cuda.cpp src/conformer.cpp src/kernels.o $(CUDA_LDFLAGS) -o $@

clean:
	rm -f parakeet parakeet.cuda src/kernels.o
