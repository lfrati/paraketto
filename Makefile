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

.PHONY: run bench bench-cpp engines inspect-onnx weights clean

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

# ONNX inspection and weight export
inspect-onnx:
	uv run python scripts/inspect_onnx.py

weights: weights.bin

weights.bin: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/cutlass_kernels.o: src/cutlass_kernels.cu src/cutlass_kernels.h
	$(NVCC) -std=c++17 -O3 -I$(CUDA_HOME)/include -Isrc -Ithird_party/cutlass/include --expt-relaxed-constexpr -gencode arch=compute_80,code=compute_80 -c $< -o $@

# TRT backend (reference)
parakeet: src/parakeet.cpp src/kernels.o src/kernels.h
	$(CXX) $(CXXFLAGS) src/parakeet.cpp src/kernels.o $(LDFLAGS) -o $@

# CUDA backend (no TensorRT dependency)
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -I/usr/include/x86_64-linux-gnu -Ithird_party/cudnn-frontend/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcufft -lcublas -lcublasLt -lcudnn -lpthread

parakeet_cuda: src/parakeet_cuda.cpp src/conformer.cpp src/conformer.h src/kernels.o src/kernels.h
	$(CXX) $(CUDA_CXXFLAGS) src/parakeet_cuda.cpp src/conformer.cpp src/kernels.o $(CUDA_LDFLAGS) -o $@

clean:
	rm -f parakeet parakeet_cuda src/kernels.o
