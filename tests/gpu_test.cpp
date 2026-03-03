// gpu_test.cpp — Smoke test for direct GPU init via ioctls (no CUDA)
//
// Exercises the full init sequence from gpu.h:
//   1. Open /dev/nvidiactl, /dev/nvidia-uvm, /dev/nvidia0
//   2. Create RM client, init UVM, enumerate GPUs
//   3. Create device/subdevice, VA space, usermode MMIO
//   4. Create channel group + compute/copy GPFIFO channels
//   5. Print GPU UUID + work submit tokens

#include "gpu.h"
#include <cstdio>

int main() {
    fprintf(stderr, "=== gpu_test: Direct GPU init (no CUDA) ===\n\n");

    GPU gpu;
    if (!gpu.init()) {
        fprintf(stderr, "\nFAILED: GPU init returned false\n");
        return 1;
    }

    fprintf(stderr, "\n=== GPU Init Summary ===\n");
    fprintf(stderr, "  GPU ID:         0x%x\n", gpu.gpu_id);
    fprintf(stderr, "  Minor:          %d\n", gpu.minor_number);
    fprintf(stderr, "  UUID:           ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", gpu.gpu_uuid.uuid[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Compute token:  0x%x\n", gpu.compute_token);
    fprintf(stderr, "  Copy token:     0x%x\n", gpu.copy_token);
    fprintf(stderr, "  GPFIFO GPU VA:  0x%lx\n", (unsigned long)gpu.gpfifo_gpu);
    fprintf(stderr, "  CmdQ GPU VA:    0x%lx\n", (unsigned long)gpu.cmdq_gpu);
    fprintf(stderr, "  MMIO mapped:    %s\n", gpu.gpu_mmio ? "yes" : "no");

    // ===== Memory test =====
    fprintf(stderr, "\n=== Memory Test ===\n");
    const uint64_t test_size = 1 << 20; // 1MB
    auto mem = gpu.gpu_malloc(test_size);
    if (!mem.cpu_ptr || !mem.gpu_addr) {
        fprintf(stderr, "FAILED: gpu_malloc returned null\n");
        return 1;
    }
    fprintf(stderr, "  Allocated 1MB: gpu_va=0x%lx cpu=%p\n",
            (unsigned long)mem.gpu_addr, mem.cpu_ptr);

    // Write pattern through CPU mapping
    uint32_t* p = (uint32_t*)mem.cpu_ptr;
    uint32_t n_words = test_size / 4;
    for (uint32_t i = 0; i < n_words; i++)
        p[i] = 0xDEAD0000 | (i & 0xFFFF);

    // Read back and verify
    __sync_synchronize();
    int mismatches = 0;
    for (uint32_t i = 0; i < n_words; i++) {
        uint32_t expected = 0xDEAD0000 | (i & 0xFFFF);
        if (p[i] != expected) {
            if (mismatches < 5)
                fprintf(stderr, "  MISMATCH at word %u: got 0x%08x expected 0x%08x\n",
                        i, p[i], expected);
            mismatches++;
        }
    }
    if (mismatches == 0)
        fprintf(stderr, "  Memory write/read: %u words verified OK\n", n_words);
    else {
        fprintf(stderr, "  FAILED: %d mismatches out of %u words\n", mismatches, n_words);
        return 1;
    }

    fprintf(stderr, "\n=== ALL PASS ===\n");
    return 0;
}
