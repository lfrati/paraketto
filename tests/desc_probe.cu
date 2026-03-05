// Minimal kernel to see how SM120 accesses global memory
// Compile: nvcc -arch=sm_120 -cubin desc_probe.cu -o desc_probe.cubin
#include <cstdint>
__global__ void probe(float* in, float* out, int n) {
    int i = threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}
