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
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lpthread $(TRT_LIBS)/libnvinfer.so.10 -Wl,-rpath,$(TRT_LIBS)

.PHONY: bench-all bench-py bench-cpp bench-cuda bench-cublas engines inspect-onnx weights download-data download-weights clean

# Download benchmark data from GitHub release
data/librispeech/manifest.json:
	@echo "Downloading benchmark data..."
	@gh release download bench-data --pattern 'bench-data.tar.gz' && \
		tar xzf bench-data.tar.gz && \
		rm bench-data.tar.gz && \
		echo "Downloaded $$(find data/ -name '*.wav' | wc -l) wav files"

download-data: data/librispeech/manifest.json

# Download pre-exported weights from GitHub release
weights.bin:
	@echo "Downloading weights..."
	@gh release download bench-data --pattern 'weights.bin'
	@echo "Downloaded weights.bin ($$(du -h weights.bin | cut -f1))"

download-weights: weights.bin

engines: engines/encoder.engine engines/decoder_joint.engine

engines/encoder.engine engines/decoder_joint.engine: scripts/build_engines.py
	uv run python scripts/build_engines.py

BENCH_SEP = @printf '\n%s\n%s\n%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' '  $(1)'

bench-all: paraketto engines/encoder.engine engines/decoder_joint.engine paraketto.cuda data/librispeech/manifest.json weights.bin
	$(call BENCH_SEP,Python  ·  ONNX Runtime + TRT EP)
	@uv run python tests/bench.py
	$(call BENCH_SEP,C++ TRT ·  paraketto.cpp + TensorRT)
	@uv run python tests/bench_cpp.py
	$(call BENCH_SEP,C++ CUDA · paraketto_cuda.cpp + CUTLASS)
	@uv run python tests/bench_cuda.py

bench-py: data/librispeech/manifest.json
	uv run python tests/bench.py

bench-cpp: paraketto engines/encoder.engine engines/decoder_joint.engine data/librispeech/manifest.json
	uv run python tests/bench_cpp.py

bench-cuda: paraketto.cuda data/librispeech/manifest.json weights.bin
	uv run python tests/bench_cuda.py paraketto.cuda

bench-cublas: paraketto.cublas data/librispeech/manifest.json weights.bin
	uv run python tests/bench_cuda.py paraketto.cublas

# ONNX inspection and weight export
inspect-onnx:
	uv run python scripts/inspect_onnx.py

weights: weights.bin

# Re-generate weights from ONNX (only needed if export script changes)
weights-export: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

SHARED_HEADERS = src/common.h src/wav.h src/mel.h src/vocab.h src/server.h

# TRT backend (reference)
paraketto: src/paraketto.cpp src/kernels.o src/kernels.h $(SHARED_HEADERS)
	$(CXX) $(CXXFLAGS) src/paraketto.cpp src/kernels.o $(LDFLAGS) -o $@

# Shared CUDA backend flags
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread
CUTLASS_INC   = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include
CONFORMER_DEPS = src/paraketto_cuda.cpp src/conformer.cpp src/conformer.h src/kernels.h src/gemm.h $(SHARED_HEADERS)

# CUTLASS GEMM backend (default — no cuBLAS dependency, cudart only)
src/cutlass_gemm.o: src/cutlass_gemm.cu src/cutlass_gemm.h src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_80 $(CUTLASS_INC) -c $< -o $@

paraketto.cuda: $(CONFORMER_DEPS) src/kernels.o src/cutlass_gemm.o src/cutlass_gemm.h
	$(CXX) $(CUDA_CXXFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/kernels.o src/cutlass_gemm.o $(CUDA_LDFLAGS) -o $@

# cuBLAS GEMM backend (faster on some shapes, requires libcublas)
src/cublas_gemm.o: src/cublas_gemm.cu src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

paraketto.cublas: $(CONFORMER_DEPS) src/kernels.o src/cublas_gemm.o
	$(CXX) $(CUDA_CXXFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/kernels.o src/cublas_gemm.o $(CUDA_LDFLAGS) -lcublas -lcublasLt -o $@

clean:
	rm -f paraketto paraketto.cuda paraketto.cublas src/kernels.o src/cutlass_gemm.o src/cublas_gemm.o
