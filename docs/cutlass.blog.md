# Freedom: Eliminating CUDA from GPU Inference

## The Problem

Parakeet transcribes 10 seconds of audio in 8 milliseconds. But before it can
transcribe anything, it spends 370 milliseconds initializing CUDA — 45x longer than
the work itself. This forces us to keep a persistent server process running,
permanently consuming ~500MB of GPU memory even when nobody is talking.

What if we could just... not load CUDA?

## The Idea

When you call `cudaMalloc` or `cudaLaunchKernel`, your code talks to `libcuda.so`
(NVIDIA's userspace driver library, ~200MB). This library creates a GPU "context" —
page tables, command queues, memory pools — and then sends work to the GPU through
the kernel driver (`nvidia.ko`).

But here's the thing: `libcuda.so` isn't magic. It just calls `ioctl()` on
`/dev/nvidiactl` and writes to memory-mapped GPU registers. The GPU hardware has a
ring buffer called the GPFIFO — you write commands into it, poke a "doorbell"
register, and the GPU's DMA engine reads and executes them. Anyone with access to
the device files can do this.

[tinygrad](https://github.com/tinygrad/tinygrad) proved it works. Their NV backend
(`ops_nv.py`) bypasses every NVIDIA userspace library. It opens `/dev/nvidiactl`,
creates GPU objects via ioctls, and pushes pre-compiled kernels into the GPFIFO
directly. The result: near-zero launch latency and no CUDA initialization cost.

**Our plan**: Replace cuBLAS with pre-compiled CUTLASS kernels (compiled to GPU
machine code ahead of time), then replace the CUDA runtime with direct ioctls. The
result would be a single ~1MB binary that opens device files, pushes commands, and
starts transcribing. Cold start: ~25ms instead of 370ms.

## Step 1: What Is cuBLAS Actually Doing?

Before replacing anything, we need to know exactly what cuBLAS runs on our GPU.
Neural network inference is dominated by matrix multiplications (GEMMs), and cuBLAS
is NVIDIA's library for these. But cuBLAS is a black box — it selects kernels
internally based on matrix shapes and GPU architecture.

We used two approaches to peek inside:

### Instrumenting the heuristic

cuBLASLt (the "light" API) exposes an algorithm selection step. When you ask it to
multiply two matrices, it first queries a heuristic that returns an algorithm ID,
tile configuration, and pipeline depth. We added logging to dump these:

```
GEMM NN  4096x 63x1024: algo=21 tile=64x64  stages=6  splitk=1    ← encoder FF layers
GEMM NN  3072x 63x1024: algo=67 tile=64x64  stages=AUTO           ← fused QKV projection
GEMM NT  2048x 63x1024: algo=21 tile=32x32  stages=2              ← conv pointwise
GEMM NT  2560x  1x1280: algo=21 tile=16x16  stages=1  bias        ← decoder LSTM
GEMM NN   640x  1x 640: algo=21 tile=16x16  stages=1  bias        ← decoder joint
```

The "tile" is the chunk of the output matrix each thread block computes. The
"stages" control how many tiles of input data are being loaded concurrently
(pipelining memory loads with computation). "splitk" means the K dimension is
split across multiple thread blocks, requiring a reduction pass.

### Profiling with nsight systems

The real revelation came from nsight systems, which captures the actual kernel names:

```
28.0%  cutlass_80_tensorop_h16816gemm_64x64_64x6_nn_align8     (196 calls, 12us avg)
10.2%  nvjet_sm120_hhh_mma_64x64x128_tmaAB_bz_NNNN            (96 calls, 9us avg)
 8.9%  cutlass_80_wmma_h161616gemm_16x16_128x1_tn_align8       (132 calls, 5.7us avg)
 7.6%  cutlass_80_wmma_h161616gemm_32x32_128x2_tn_align8       (96 calls, 6.6us avg)
 7.5%  magma_hgemmEx_kernel                                     (96 calls, 6.6us avg)
 4.7%  cutlass_80_wmma_h161616gemm_16x16_32x1_nn_align8        (66 calls, 6.0us avg)
 4.1%  cutlass_80_tensorop_h16816gemm_64x64_32x6_nn_align2     (66 calls, 5.2us avg)
 4.1%  cutlass_80_wmma_h161616gemm_32x32_128x2_nn_align8       (48 calls, 7.1us avg)
```

### The surprise

**cuBLAS is running Ampere-era (SM80) CUTLASS kernels on our Blackwell GPU.**

Every kernel name starting with `cutlass_80_` is an SM80 template — designed for
the RTX 3090 generation. Our RTX 5070 Ti (SM120, Blackwell GeForce) has new tensor
core instructions, but only for narrow-precision formats (FP8/FP6/FP4). For FP16,
SM120's tensor cores use the same `mma.sync` instruction as Ampere. So even NVIDIA
falls back to Ampere kernels.

There's exactly one native SM120 kernel in the whole profile: `nvjet_sm120_*`,
accounting for 10% of GEMM time. This is a newer "nvjet" framework kernel that uses
SM120's TMA (Tensor Memory Accelerator) for data loading. It's used for the fused
QKV projection and position encoding — shapes where the heuristic selected algo=67
instead of algo=21.

The remaining 7.5% uses MAGMA kernels for batched (multi-head attention) GEMMs.

### What this means

Since cuBLAS is already using CUTLASS SM80 templates, **we can compile the exact
same CUTLASS templates ourselves**. We know the tile sizes, pipeline depths, and
layouts for every shape. We don't need to reverse-engineer proprietary kernels or
write hand-tuned assembly — we just instantiate the same open-source templates that
cuBLAS is using.

The only question marks:
- The `nvjet_sm120` kernel (10% of time) — we'll cover this with SM80 CUTLASS and
  benchmark the difference
- The MAGMA batched kernels (7.5%) — we need our own strided batched GEMM
- The decoder GEMV (M=1 matrix-vector products) — cuBLAS wastes 16x16 tiles on
  what should be a simple reduction. A custom kernel will be faster here.

### The CUTLASS templates we need

| Template | Tile | Pipeline | Layout | Shapes |
|----------|------|----------|--------|--------|
| `tensorop_h16816gemm` | 64x64 | K=64, 6 stages | NN | FF1/FF2 (28% of time) |
| `tensorop_h16816gemm` | 64x64 | K=32, 6 stages | NN | Encoder projection |
| `wmma_h161616gemm` | 32x32 | K=128, 2 stages | TN | Conv pointwise (NT layout) |
| `wmma_h161616gemm` | 32x32 | K=128, 2 stages | NN | Position encoding |
| `wmma_h161616gemm` | 16x16 | K=128, 1 stage | TN | Decoder LSTM GEMV |
| `wmma_h161616gemm` | 16x16 | K=32, 1 stage | NN | Decoder joint GEMV |
| `tensorop_h16816gemm` | 128x128 | K=32, 5 stages | NN | Subsampling (large) |
| custom GEMV | — | — | both | Decoder M=1 (potential speedup) |
| batched GEMM | — | — | both | Multi-head attention |

The total number of unique kernel configurations is small — roughly 10 cubins
covering all ~1000 kernel launches per inference pass.

## Step 2: Writing the CUTLASS Kernels

We instantiated CUTLASS SM80 FP16 GEMM templates and benchmarked them against
cuBLAS for every shape in the model. Since cuBLAS is already running CUTLASS SM80
kernels internally, the question was whether we could match its algorithm selection.

### The configs we used

All kernels use `arch::Sm80` with `OpClassTensorOp` and `mma.sync 16x8x16`:

| Config | Tile | K/stage | Stages | Layout | Align |
|--------|------|---------|--------|--------|-------|
| `64x64_64_s6` | 64x64 | 64 | 6 | NN + TN | 8 |
| `64x64_32_s6` | 64x64 | 32 | 6 | NN | 8 |
| `64x64_32_s3` | 64x64 | 32 | 3 | NN | 8 |
| `64x64_64_s6_sk4` | 64x64 | 64 | 6 | NN | 8 | split-K=4 |
| `64x64_32_s6_a2` | 64x64 | 32 | 6 | NN | 2 | for M=1030 |
| `128x128_32_s5` | 128x128 | 32 | 5 | NN | 8 |
| `128x64_32_s3` | 128x64 | 32 | 3 | TN | 8 |

### Results: CUTLASS matches or beats cuBLAS

```
Shape                cuBLAS(us)  CUTLASS(us)  Config           Ratio   Notes
FF1 linear1 (NN)         8.2         8.2      64x64_64_s6      1.00x   Matched
FF1 linear2 (NN)        10.3        12.3      64x64_64_sk2     1.20x   Split-K gap (only slow one)
Fused QKV (NN)           8.0         6.2      64x64_32_s10     0.78x   CUTLASS 22% faster!
Pos enc proj (NN)        6.0         6.2      64x64_32_s6      1.03x   ~Matched
MHSA out (NN)            6.2         6.2      64x64_64_s6      1.00x   Matched
Enc proj (NN)            6.2         6.2      64x64_64_s6      1.00x   Matched
Conv pw1 (TN)            8.2         6.2      64x64_64_s6      0.75x   CUTLASS 25% faster!
Conv pw2 (TN)            6.2         6.2      64x64_64_s6      1.00x   Matched
LSTM gates (TN)          6.2         6.2      64x64_64_s6      1.00x   Matched
Dec proj (NN)            6.2         4.1      64x64_32_s3      0.67x   CUTLASS 33% faster!
Out proj (NN)            6.2         6.2      64x64_32_s6_a2   1.00x   Matched (align2 fix)
Sub conv.3 (NN)          8.2         8.2      128x128_32_s5    1.00x   Matched
Sub conv.6 (NN)          6.2         3.9      64x64_32_s10     0.64x   CUTLASS 36% faster!
```

### Total impact on inference

Weighting by call count per inference pass (24 blocks x encoder calls, ~33
decoder steps):

| | cuBLAS | CUTLASS |
|---|---|---|
| Total GEMM time | 3,754 us | 3,508 us |
| **Delta** | | **-6.5% faster** |

CUTLASS is **6.5% faster overall** than cuBLAS, even with the 20% regression on the
one split-K shape. The decoder GEMVs (where cuBLAS wastes full GEMM tiles on M=1
vector-matrix products) are the biggest win.

### What we learned

1. **You don't need to match cuBLAS's exact tile configs.** A handful of generic
   CUTLASS templates (64x64 and 128x128 tiles with varying K and stages) cover all
   13 shapes.

2. **cuBLAS's WMMA kernels aren't special.** For the shapes where cuBLAS used WMMA
   16x16x16 tiles, our standard TensorOp 16x8x16 tiles performed the same or better.

3. **cuBLAS is suboptimal for GEMV.** The M=1 decoder shapes (matrix-vector products)
   are launched with full GEMM tiles — a custom GEMV kernel would be even faster.

4. **Split-K is the main gap.** cuBLAS's split-K=4 for the 1024x63x4096 shape
   (FF1/FF2 linear2) is 20% faster than both our serial and parallel split-K
   implementations. We tried serial split-K, parallel split-K (separate reduction
   kernel matching cuBLAS's approach), and 32x32 tiles (to avoid split-K entirely
   by creating more thread blocks). None closed the gap — cuBLAS likely has a more
   optimized mainloop or reduction kernel for this specific shape. This is one shape
   out of 13 and accounts for +96us per inference — acceptable for now.

5. **cuBLAS's nvjet_sm120 kernel doesn't help much.** The shapes where cuBLAS used
   its native SM120 kernel (fused QKV, pos encoding) are ones where our SM80 CUTLASS
   is actually *faster*.

6. **SM80 vs SM120a compilation doesn't matter.** We tested compiling the same CUTLASS
   templates with `-arch=sm_80` (JIT to SM120 at runtime) vs `-arch=sm_120a` (native
   SM120 SASS). No difference in the critical split-K path. Minor improvements on some
   non-split-K configs, but they weren't the winners anyway.

### TODO: close the split-K gap

The FF1 linear2 shape (1024x63x4096) is the one remaining regression. Potential
approaches for later:
- Profile with `ncu` to compare instruction-level differences between cuBLAS and
  CUTLASS split-K kernels
- Try CUTLASS 3.x `CollectiveBuilder` API which has a different mainloop architecture
- Write a custom split-K reduction kernel optimized for this specific shape
- Try a hand-tuned GEMV-style kernel since N=63 is very small

### Next: batched GEMMs for attention

The remaining cuBLAS dependency is the strided batched GEMM for multi-head attention
(cuBLAS dispatches MAGMA kernels for these, 7.5% of GPU time). This needs either
CUTLASS batched GEMM or a loop over regular GEMMs.

*[Step 3: Building the ioctl GPU launcher — next]*
