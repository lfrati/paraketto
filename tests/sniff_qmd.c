// sniff_qmd.c — LD_PRELOAD hook to intercept CUDA kernel launches and dump QMD
// Usage: LD_PRELOAD=./sniff_qmd.so ./test_cuda_verify
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

// We intercept writes to the GPFIFO doorbell (MMIO offset 0x90)
// and the GPFIFO ring buffer writes to find the QMD address

// Track mmap'd regions to find GPFIFO control areas
static void* mmio_base = NULL;
static size_t mmio_size = 0;

// Look for SEND_PCAS pattern in command buffers
// The command format is: (1 << 29) | (count << 16) | (subchannel << 13) | (method >> 2)
// SEND_PCAS_A method = 0x02B4, subchannel = 1
// So: 0x20012000 | (0x02B4 >> 2) = 0x200120AD

// Instead, let's just scan all mmap'd memory for QMD patterns
// A QMD v5 has: major_version=5 at bits 471:468 -> DW[14] bits 23:20 = 5

static int (*real_ioctl)(int fd, unsigned long request, ...) = NULL;

// Hook mmap to track GPU memory mappings
void* (*real_mmap)(void*, size_t, int, int, int, off_t) = NULL;

// Alternative approach: just intercept the write to doorbell
// The GPFIFO ring entry points to the command buffer which has SEND_PCAS
// The SEND_PCAS_A data contains (qmd_addr >> 8)
// So we need to find the QMD in GPU memory

// Simplest approach: after the CUDA launch, scan all mapped memory for QMD v5 signature
// QMD v5: dw[14] bits 23:20 = 5 (qmd_major_version)
// And dw[4] bit 24:20 has qmd_type etc

// Actually, let's just take a different approach entirely.
// Write the cbuf0 descriptor dump as a PTX kernel using proper syntax.

int main() { return 0; }  // unused
