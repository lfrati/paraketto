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

.PHONY: bench-all bench-py bench-cpp bench-cuda bench-cublas bench-fp8 engines inspect-onnx weights weights-fp8 download-data download-weights clean

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

bench-all: paraketto engines/encoder.engine engines/decoder_joint.engine paraketto.cuda paraketto.cublas paraketto.fp8 data/librispeech/manifest.json weights.bin
	$(call BENCH_SEP,Python  ·  ONNX Runtime + TRT EP)
	@uv run python tests/bench.py
	$(call BENCH_SEP,C++ TRT ·  paraketto.cpp + TensorRT)
	@uv run python tests/bench_cpp.py
	$(call BENCH_SEP,C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16)
	@uv run python tests/bench_native.py paraketto.cuda
	$(call BENCH_SEP,C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16)
	@uv run python tests/bench_native.py paraketto.cublas
	$(call BENCH_SEP,C++ FP8  · paraketto_cuda.cpp + cublasLt FP8)
	@uv run python tests/bench_native.py paraketto.fp8

bench-py: data/librispeech/manifest.json
	uv run python tests/bench.py

bench-cpp: paraketto engines/encoder.engine engines/decoder_joint.engine data/librispeech/manifest.json
	uv run python tests/bench_cpp.py

bench-cuda: paraketto.cuda data/librispeech/manifest.json weights.bin
	uv run python tests/bench_native.py paraketto.cuda

bench-cublas: paraketto.cublas data/librispeech/manifest.json weights.bin
	uv run python tests/bench_native.py paraketto.cublas

bench-fp8: paraketto.fp8 data/librispeech/manifest.json weights_fp8.bin
	uv run python tests/bench_native.py paraketto.fp8

# ONNX inspection and weight export
inspect-onnx:
	uv run python scripts/inspect_onnx.py

weights: weights.bin

# Re-generate weights from ONNX (only needed if export script changes)
weights-export: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 -c $< -o $@

SHARED_HEADERS = src/common.h src/wav.h src/mel.h src/vocab.h src/server.h

# TRT backend (reference)
paraketto: src/paraketto.cpp src/kernels.o src/kernels.h $(SHARED_HEADERS)
	$(CXX) $(CXXFLAGS) src/paraketto.cpp src/kernels.o $(LDFLAGS) -o $@

# Shared CUDA backend flags
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread
CUTLASS_INC   = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include
src/weights.o: src/weights.cpp src/conformer.h src/common.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

CONFORMER_DEPS = src/paraketto_cuda.cpp src/conformer.cpp src/conformer.h src/kernels.h src/gemm.h $(SHARED_HEADERS)

# CUTLASS GEMM backend (default — no cuBLAS dependency, cudart only)
src/cutlass_gemm.o: src/cutlass_gemm.cu src/cutlass_gemm.h src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) -c $< -o $@

paraketto.cuda: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cutlass_gemm.o src/cutlass_gemm.h
	$(CXX) $(CUDA_CXXFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cutlass_gemm.o $(CUDA_LDFLAGS) -o $@

# cuBLAS GEMM backend (faster on some shapes, requires libcublas)
src/cublas_gemm.o: src/cublas_gemm.cu src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 -c $< -o $@

paraketto.cublas: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cublas_gemm.o
	$(CXX) $(CUDA_CXXFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cublas_gemm.o $(CUDA_LDFLAGS) -lcublas -lcublasLt -o $@

# FP8 cublasLt backend
src/kernels_fp8.o: src/kernels_fp8.cu src/kernels_fp8.h
	$(NVCC) $(NVFLAGS) -arch=sm_120a -c $< -o $@

src/conformer_fp8.o: src/conformer_fp8.cpp src/conformer_fp8.h src/conformer.h src/kernels.h src/kernels_fp8.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

paraketto.fp8: src/paraketto_cuda.cpp src/conformer_fp8.h src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o $(SHARED_HEADERS)
	$(CXX) $(CUDA_CXXFLAGS) -include src/conformer_fp8.h src/paraketto_cuda.cpp src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o $(CUDA_LDFLAGS) -lcublas -lcublasLt -o $@

# Generate FP8 weight cache (quantized from weights.bin, auto-saved by paraketto.fp8 on first run)
weights_fp8.bin: paraketto.fp8 weights.bin
	@echo "Generating FP8 weight cache (quantizing weights.bin → weights_fp8.bin)..."
	@rm -f weights_fp8.bin
	@./paraketto.fp8 --weights weights.bin /dev/null 2>&1 | grep -E "saved|loaded|FP8" || true
	@test -f weights_fp8.bin || (echo "ERROR: weights_fp8.bin not created" && exit 1)

weights-fp8: weights_fp8.bin

# Convert existing weight files to current format (run once after updating)
repack: weights.bin
	uv run python scripts/repack_weights.py

# Static build with embedded weights (single file, only needs NVIDIA driver)
weights_embedded.o: weights.bin
	objcopy -I binary -O elf64-x86-64 -B i386:x86-64 \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		$< $@

weights_fp8_embedded.o: weights_fp8.bin
	objcopy -I binary -O elf64-x86-64 -B i386:x86-64 \
		--rename-section .data=.rodata,alloc,load,readonly,data,contents \
		$< $@

paraketto.static: $(CONFORMER_DEPS) src/kernels.o src/cutlass_gemm.o src/cutlass_gemm.h weights_embedded.o
	$(CXX) $(CUDA_CXXFLAGS) -DEMBEDDED_WEIGHTS \
		src/paraketto_cuda.cpp src/conformer.cpp src/kernels.o src/cutlass_gemm.o weights_embedded.o \
		-static-libstdc++ -static-libgcc \
		-L$(CUDA_HOME)/lib64 $(CUDA_HOME)/lib64/libcudart_static.a -ldl -lpthread -lrt \
		-o $@

# FP8 static: embeds only weights_fp8.bin — no weights.bin needed at runtime
paraketto.fp8.static: src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o weights_fp8_embedded.o $(SHARED_HEADERS) src/conformer_fp8.h
	$(CXX) $(CUDA_CXXFLAGS) -DEMBEDDED_WEIGHTS -include src/conformer_fp8.h \
		src/paraketto_cuda.cpp src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o \
		weights_fp8_embedded.o \
		-static-libstdc++ -static-libgcc \
		-L$(CUDA_HOME)/lib64 $(CUDA_HOME)/lib64/libcudart_static.a -ldl -lpthread -lrt \
		-lcublas -lcublasLt \
		-o $@

bench_gemm: tests/bench_gemm.cu
	$(NVCC) $(NVFLAGS) -arch=sm_80 $(CUTLASS_INC) tests/bench_gemm.cu -lcublas -lcublasLt -o $@

bench_splitk: tests/bench_splitk.cu
	$(NVCC) $(NVFLAGS) -arch=sm_80 $(CUTLASS_INC) tests/bench_splitk.cu -lcublas -lcublasLt -o $@

bench_tiles: tests/bench_tiles.cu
	$(NVCC) $(NVFLAGS) -arch=sm_80 $(CUTLASS_INC) tests/bench_tiles.cu -lcublas -lcublasLt -o $@

bench_ff2: tests/bench_ff2.cu
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) tests/bench_ff2.cu -lcublas -lcublasLt -o $@

clean:
	rm -f paraketto paraketto.cuda paraketto.cublas paraketto.fp8 paraketto.static paraketto.fp8.static src/kernels.o src/kernels_fp8.o src/cutlass_gemm.o src/cublas_gemm.o src/weights.o src/conformer_fp8.o weights_embedded.o weights_fp8_embedded.o
