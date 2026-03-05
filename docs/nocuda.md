# nocuda: Launching GPU Kernels Without libcuda.so

**Target**: NVIDIA RTX 5070 Ti — SM120 (Blackwell), CUDA 13.1, driver 590.48.01
**Status**: Kernel launch + semaphore sync fully working
**Binary dependencies**: libc, libpthread, libm — zero NVIDIA userspace libs

---

## 1. What This Is

A from-scratch GPU driver that talks directly to the NVIDIA kernel module
(`/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm`) via ioctls, bypassing
libcuda.so, libcudart.so, and libcublas.so entirely. Kernels are compiled to
`.cubin` with nvcc at build time, then loaded and launched at runtime through
raw GPFIFO command submission.

The approach follows tinygrad's `ops_nv.py` for the init sequence, and uses
NVIDIA's MIT-licensed `open-gpu-kernel-modules` headers for all struct
definitions and constants.

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│  test_launch.cpp                                      │
│  - loads kernels.cubin via cubin_loader.h              │
│  - fills input buffers, calls gpu.launch_kernel()      │
│  - waits on semaphore, verifies output                 │
├──────────────────────────────────────────────────────┤
│  gpu.h — single-header ioctl driver                    │
│  ┌─────────────┬──────────────┬─────────────────────┐ │
│  │ init()      │ gpu_malloc() │ launch_kernel()     │ │
│  │ 20 RM steps │ alloc+map    │ QMD + GPFIFO submit │ │
│  └─────────────┴──────────────┴─────────────────────┘ │
├──────────────────────────────────────────────────────┤
│  cubin_loader.h — ELF parser                           │
│  - .text.<name> → code offset/size                     │
│  - .nv.info.<name> → reg_count, param_base             │
│  - .nv.constant0.<name> → cbuf0 total size             │
│  - .nv.shared.<name> → shared memory size              │
├──────────────────────────────────────────────────────┤
│  /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm         │
│  (NVIDIA kernel module, no userspace libs)             │
└──────────────────────────────────────────────────────┘
```

## 3. Init Sequence (20 Steps)

The GPU initialization follows tinygrad's `ops_nv.py` exactly. Each step is
an ioctl to `/dev/nvidiactl` (RM API) or `/dev/nvidia-uvm` (UVM API):

| Step | What | ioctl | Key params |
|------|------|-------|------------|
| 1 | Open device files | `open()` | `/dev/nvidiactl`, `/dev/nvidia-uvm` (x2), `/dev/nvidia0` |
| 2 | Create RM root client | `NV_ESC_RM_ALLOC` | class=`NV01_ROOT_CLIENT` |
| 3 | Init UVM | `UVM_INITIALIZE` + `UVM_MM_INITIALIZE` | second UVM fd for MM |
| 4 | Enumerate GPUs | `NV_ESC_CARD_INFO` | returns gpu_id, minor_number |
| 5 | Open GPU device | `open("/dev/nvidia0")` + `NV_ESC_REGISTER_FD` | |
| 6 | Create device | `NV_ESC_RM_ALLOC` | class=`NV01_DEVICE_0`, vaMode=MULTIPLE_VASPACES |
| 7 | Create subdevice | `NV_ESC_RM_ALLOC` | class=`NV20_SUBDEVICE_0` |
| 8 | Create virtual memory handle | `NV_ESC_RM_ALLOC` | class=`NV01_MEMORY_VIRTUAL`, limit=0x1FFFFFFFFFFFF |
| 9 | Map usermode MMIO | `NV_ESC_RM_ALLOC` + `mmap` | class=`HOPPER_USERMODE_A`, 64KB region for doorbell |
| 10 | Boost clocks | `NV_ESC_RM_CONTROL` | `NV2080_CTRL_CMD_PERF_BOOST`, duration=0xFFFFFFFF |
| 11 | Get GPU UUID | `NV_ESC_RM_CONTROL` | `NV2080_CTRL_CMD_GPU_GET_GID_INFO`, 16-byte binary UUID |
| 12 | Create VA space | `NV_ESC_RM_ALLOC` | class=`FERMI_VASPACE_A`, flags=PAGE_FAULTING+EXTERNALLY_OWNED |
| 13 | Register with UVM | `UVM_REGISTER_GPU` + `UVM_REGISTER_GPU_VASPACE` | links RM client to UVM |
| 14 | Create channel group (TSG) | `NV_ESC_RM_ALLOC` | class=`KEPLER_CHANNEL_GROUP_A`, engine=GRAPHICS |
| 15 | Allocate GPFIFO memory | `NV_ESC_RM_ALLOC` | 3MB contiguous write-combined, class=`NV1_MEMORY_USER` |
| 16 | Create context share | `NV_ESC_RM_ALLOC` | class=`FERMI_CONTEXT_SHARE_A`, flags=SUBCONTEXT_ASYNC |
| 17 | Create compute channel | `NV_ESC_RM_ALLOC` | class=`BLACKWELL_CHANNEL_GPFIFO_A` (0xC96F) |
| 18 | Create copy channel | `NV_ESC_RM_ALLOC` | class=`BLACKWELL_CHANNEL_GPFIFO_A` (separate ring offset) |
| 19 | Schedule channel group | `NV_ESC_RM_CONTROL` | `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`, bEnable=1 |
| 20 | Allocate command queue | `NV_ESC_RM_ALLOC` | 2MB for pushbuffer methods |

Each channel also gets a work submit token via `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN`
and is registered with UVM via `UVM_REGISTER_CHANNEL`.

## 4. Memory Allocation

Every GPU buffer goes through 6 operations:

```
1. alloc_va()                    — bump-allocate a GPU virtual address (starts at 0x1000000000)
2. NV_ESC_RM_ALLOC (NV1_MEMORY_USER) — allocate physical GPU memory
3. NV_ESC_RM_MAP_MEMORY (mmap)   — map to CPU address space (fresh fd per mapping)
4. UVM_CREATE_EXTERNAL_RANGE     — tell UVM about this VA range
5. NV_ESC_RM_MAP_MEMORY_DMA      — map physical pages into channel VA space
6. UVM_MAP_EXTERNAL_ALLOCATION   — map physical pages into UVM VA space
```

Steps 5 and 6 create redundant mappings: the channel page tables (used by the
HOST/FIFO engine) and UVM page tables (used by the SM/compute engine) both map
the same virtual address to the same physical memory. This dual mapping is
required because the HOST engine (semaphore writes) and SM (kernel execution)
use different address translation paths.

## 5. CUBIN Loading

nvcc compiles CUDA kernels to a `.cubin` file — a standard ELF binary with
CUDA-specific sections. At runtime, we parse the ELF to extract:

| Section | What we extract |
|---------|----------------|
| `.text.<mangled_name>` | SASS code: offset within ELF, size |
| `.nv.info.<mangled_name>` | `EIATTR_PARAM_CBANK` (0x0A) → param_base offset in cbuf0 |
| `.nv.info` (global) | `EIATTR_REGCOUNT` (0x2F) → register count per kernel (Blackwell stores these globally, indexed by symbol) |
| `.nv.constant0.<mangled_name>` | Section size → total cbuf0 size (Blackwell min is 924 bytes) |
| `.nv.shared.<mangled_name>` | Section size → shared memory bytes |

The entire ELF image is uploaded contiguously to GPU memory. Kernel addresses
are `cubin_base + code_offset`.

### .nv.info format

```
while offset < section_size:
    fmt   = u8 at offset+0     (0x04 = variable-length, else fixed 4-byte)
    param = u8 at offset+1     (EIATTR type)
    val16 = u16 at offset+2    (size if fmt==0x04, else the data itself)
    if fmt == 0x04:
        payload = bytes[offset+4 : offset+4+val16]
        offset += 4 + val16
    else:
        data = val16
        offset += 4
```

## 6. QMD v5 (Queue Meta Data) — Blackwell

The QMD is a 384-byte (96 DWORD) descriptor that tells the GPU everything
about a kernel launch. Blackwell uses QMD major version 5.

### What worked: captured CUDA template

We launched a kernel via CUDA and scanned process memory for the QMD
(searching for the v5 signature: `DW[14] bits 23:20 = 5`, SASS version
`DW[14] bits 7:0 = 0xA4`). This gave us the exact DWORD values CUDA uses.

### Fixed DWORDs (captured from CUDA, used as-is)

```
DW[ 4] = 0x013f0000    qmd_type=2 (GRID_CTA), qmd_group_id=0x3F
DW[ 9] = 0x00000000    QMD release disabled (set by wait_kernel when needed)
DW[10] = 0x00000000    DEPENDENT_QMD0 fields (set by chaining code, not template)
DW[12] = 0x00000000    DEPENDENT_QMD0_POINTER (set by chaining code, not template)
DW[14] = 0x2f5003a4    sass_version=0xA4, sampler_index=1, qmd_major_version=5
DW[19] = 0x80610000    cwd_membar_type=1 + flags
DW[20] = 0x00000008    (unknown, but required)
DW[22] = 0x04000000    (unknown, but required)
DW[58] = 0x00000011    constant buffer valid bits: cbuf0 + cbuf1
```

### Variable DWORDs (patched per launch)

```
DW[15:16]  = semaphore GPU virtual address (low, high)
DW[17:18]  = semaphore payload (counter value, 0)
DW[32:33]  = program address >> 4 (low, high 21 bits)
DW[34]     = (block_y << 16) | block_x
DW[35]     = block_z | (reg_count << 8) | (barrier_count << 17)
DW[36]     = shared_mem>>7 | (MIN_SMEM_CFG=3 << 11) | (MAX=0x1A << 17) | (TARGET=5 << 23)
DW[39:41]  = grid_x, grid_y, grid_z
DW[42:43]  = cbuf0 address >> 6, (cbuf0_addr_hi | cbuf0_size>>4 << 19)
DW[59:60]  = program prefetch address >> 8
```

### What DOESN'T work (and why)

These fields exist in CUDA's QMD but cause total regression when set in ours:

| Field | CUDA value | Effect when set | Likely cause |
|-------|------------|-----------------|-------------|
| DW[33] prefetch bits (21:30) | TYPE=0, SIZE=11 | All output = 0xBEEF | Prefetch requires local memory pool we don't allocate |
| DW[37] LOCAL_MEM_HIGH | 0x00640000 (1600 bytes) | All output = 0xBEEF | We don't allocate a local memory heap via SET_SHADER_LOCAL_MEMORY |
| DW[58] = 0x09 (different cbuf valid mask) | 0x10300011 in CUDA | All output = 0xBEEF | CUDA binds cbufs 1,5,7 with specific addresses; we only have cbuf0 |
| SEND_SIGNALING_PCAS2_B != 9 | value 1 | Kernel never dispatches | Action 9 = PREFETCH_SCHEDULE, the only valid dispatch action |

**Lesson**: The QMD is extremely sensitive. Any field that references resources
we haven't allocated (local memory, additional constant buffers, prefetch
buffers) causes the kernel to silently fail — the output buffer stays
untouched (0xBEEF pattern).

## 7. Constant Buffer 0 (cbuf0) — The Driver Template

cbuf0 is the primary constant buffer for every kernel. On Blackwell, it has a
**924-byte minimum** and is split into two regions:

```
┌─────────────────────────────────────────────┐
│ [0x000, param_base)  Driver template        │  896 bytes for add_relu
│   DW[188:189] = shared_mem_window (uint64)  │  offset 0x2F0
│   DW[190:191] = local_mem_window (uint64)   │  offset 0x2F8
│   DW[216]     = blockDim.x                  │  offset 0x360
│   DW[217]     = blockDim.y                  │  offset 0x364
│   DW[218]     = blockDim.z                  │  offset 0x368
│   DW[223]     = 0xfffdc0                    │  offset 0x37C
├─────────────────────────────────────────────┤
│ [param_base, cbuf0_size)  Kernel arguments  │  28 bytes for add_relu
│   0x380: uint64 a_ptr                       │
│   0x388: uint64 b_ptr                       │
│   0x390: uint64 y_ptr                       │
│   0x398: int32  n                           │
└─────────────────────────────────────────────┘
```

### How we discovered the layout

1. Launched add_relu via CUDA, then scanned process memory for the QMD
2. From the QMD, extracted cbuf0's GPU virtual address and size
3. Searched mapped regions for a known signature: `DW[188:189] = 0x729400000000`
   (shared memory window)
4. Verified by checking that `DW[224:230]` contained the known GPU pointers for
   a, b, y and the integer N

### The blockDim.x discovery

The kernel reads `blockDim.x` from cbuf0 at offset 0x360 via `LDC c[0x0][0x360]`.
Without this, only threads with `threadIdx.x < 256` in the first block produced
correct output (256 out of 4096 elements). The kernel was reading blockDim.x as
zero from uninitialized memory, so `blockDim.x * blockIdx.x` was always 0 for
the index calculation — every block computed the same 256 elements.

**This was the fix that made all 4096 elements correct.**

### Memory windows

```
shared_mem_window = 0x729400000000    SET_SHADER_SHARED_MEMORY_WINDOW
local_mem_window  = 0x729300000000    SET_SHADER_LOCAL_MEMORY_WINDOW
```

These are set both in cbuf0 (so the kernel can read them) and via pushbuffer
methods (so the hardware knows them). They must match.

## 8. Command Submission

### NVIDIA Method (NVM) encoding

All GPU commands are encoded as 32-bit "method" headers followed by data words:

```
bits 31:29 = SEC_OP (1 = INC_METHOD, auto-incrementing)
bits 28:16 = COUNT (number of data words following)
bits 15:13 = SUBCHANNEL (0=HOST, 1=compute, 4=DMA copy)
bits 12:0  = METHOD_ADDRESS (register byte offset >> 2)
```

C implementation:
```c
void nvm(int subchannel, uint32_t reg, uint32_t count) {
    cmd((1 << 29) | (count << 16) | (subchannel << 13) | (reg >> 2));
}
```

This is equivalent to tinygrad's `(2 << 28)` — both produce `0x20000000` for
the SEC_OP field.

### Subchannel assignment

| Subchannel | Class | Purpose |
|------------|-------|---------|
| 0 | NVC96F (implicit) | HOST methods: semaphores, TLB invalidation |
| 1 | NVCEC0 (BLACKWELL_COMPUTE_B) | Compute: kernel dispatch, memory windows |
| 4 | NVCAB5 (BLACKWELL_DMA_COPY_B) | DMA copy engine |

Subchannel 0 does **not** need SET_OBJECT — the GPFIFO class methods are
handled by the host/FIFO engine implicitly. Only subchannels 1 and 4 need
explicit SET_OBJECT binding.

### Channel setup (once per session)

```
nvm(1, 0x0000, 1); data(0xCEC0);         // SET_OBJECT: bind compute class
nvm(1, 0x07B0, 2); data(hi); data(lo);    // SET_SHADER_LOCAL_MEMORY_WINDOW
nvm(1, 0x02A0, 2); data(hi); data(lo);    // SET_SHADER_SHARED_MEMORY_WINDOW
nvm(4, 0x0000, 1); data(0xCAB5);         // SET_OBJECT: bind DMA class
```

### Kernel dispatch

```
nvm(1, 0x02B4, 1); data(qmd_gpu_addr >> 8);   // SEND_PCAS_A
nvm(1, 0x02C0, 1); data(9);                    // SEND_SIGNALING_PCAS2_B (action=9)
```

Action 9 = PCAS_ACTION_PREFETCH_SCHEDULE. This is the **only** valid dispatch
action. Changing it to 1 or any other value causes the kernel to never execute.

### GPFIFO entry format

Each GPFIFO ring entry is 64 bits:

```
GP_ENTRY0 (low dword):
  bit 0      = FETCH (0=UNCONDITIONAL)
  bits 31:2  = pushbuffer GPU address bits 31:2

GP_ENTRY1 (high dword):
  bits 7:0   = pushbuffer GPU address bits 39:32
  bit 9      = LEVEL (0=MAIN, 1=SUBROUTINE)
  bits 30:10 = LENGTH (dwords in pushbuffer segment)
  bit 31     = SYNC (0=PROCEED)
```

Construction: `entry = ((addr >> 2) << 2) | ((uint64_t)n_dwords << 42)`

### Doorbell

After writing the GPFIFO entry and updating GP_PUT, ring the doorbell:

```c
volatile uint32_t* gp_put = (volatile uint32_t*)(gpfifo_cpu + ring_offset + entries*8 + 0x8C);
*gp_put = (put + 1) % entries;
__sync_synchronize();
gpu_mmio[0x90 / 4] = work_submit_token;    // MMIO doorbell at offset 0x90
```

## 9. Synchronization — The Hard Problem

This was by far the most difficult part. The kernel computed correct results
early on, but the semaphore (GPU writing a value to CPU-visible memory to
signal completion) took days of debugging.

### What didn't work

| Approach | Result | Why |
|----------|--------|-----|
| QMD release semaphore (DW[9,15-18]) | Never writes | Unknown — QMD fields look correct, CUDA uses the same pattern. Possibly requires release fence setup we don't do. |
| NVC96F SEM_EXECUTE in same pushbuffer as kernel | Never writes | HOST engine stalls when mixed with compute methods in a single GPFIFO entry. |
| NVC96F SEM_EXECUTE with RELEASE_WFI flag | Never writes | WFI (Wait For Idle) hangs indefinitely after a SEND_PCAS compute dispatch. The HOST engine never considers the compute engine "idle" after an async dispatch. |
| Semaphore in GPFIFO memory (write-combined) | Never writes | Same underlying issue — WFI hang. |
| SET_OBJECT on subchannel 0 | No effect | Not needed — subchannel 0 methods are always handled by host. |

### What works (current approach — single GPFIFO entry)

With dependent QMD chaining, everything goes in ONE pushbuffer + ONE GPFIFO
entry. The last kernel's QMD has a release semaphore, plus a HOST SEM with
RELEASE_WFI as backup:

```c
begin_commands();
// First kernel: SEND_PCAS
nvm(1, 0x02B4, 1); nvm_data(qmd0_addr >> 8);
nvm(1, 0x02C0, 1); nvm_data(9);    // SEND_SIGNALING_PCAS2_B
// Kernels 2..N: linked via dependent_qmd0 (no pushbuffer commands needed)

// wait_kernel():
// Set QMD release semaphore on last kernel
last_qmd->set_release_semaphore(0, sem_addr, counter);
last_qmd->set(336, 336, 0);  // disable dependent chaining on last kernel
// Append HOST semaphore with RELEASE_WFI
nvm(0, 0x005C, 5);
nvm_data(sem_addr_lo); nvm_data(sem_addr_hi);
nvm_data(counter_lo); nvm_data(counter_hi);
nvm_data(0x01100001);    // SEM_EXECUTE: RELEASE + RELEASE_WFI + 64BIT
submit_compute();         // single GPFIFO entry
```

**SEM_EXECUTE value = 0x01100001:**
- bit 0 = OPERATION_RELEASE (write payload to address)
- bit 20 = RELEASE_WFI (wait for all in-flight work before writing)
- bit 24 = PAYLOAD_SIZE_64BIT

### Historical note: the old 2-entry approach

Before dependent QMD chaining, each kernel was dispatched via SEND_PCAS, and
synchronization required **two separate GPFIFO entries** — HOST methods and
compute methods couldn't coexist in one pushbuffer, and RELEASE_WFI would
hang. With chaining, the single-entry approach works because the HOST SEM
fires after the entire chain completes.

### CPU polling

```c
volatile uint64_t* sem = (volatile uint64_t*)sem_alloc.cpu_ptr;
while (*sem < expected_counter) {
    __sync_synchronize();
}
```

Completes in ~140 spins for a 4096-element FP16 kernel (sub-microsecond).

## 10. Reverse Engineering Methodology

We built several CUDA-based sniffing tools that launch a kernel via the
standard CUDA runtime, then scan process memory to observe what CUDA did:

| Tool | What it does |
|------|-------------|
| `sniff_cbuf0.cu` | Finds cbuf0 in mapped memory by searching for the shared_mem_window signature (0x729400000000). Dumps the full 924-byte driver template to reveal all fixed offsets. |
| `sniff_qmd2.cu` | Scans for QMD v5 signature (version=5, sass=0xA4, type=2) and dumps all 96 DWORDs. Cross-references grid/block dims to confirm it's our kernel. |
| `sniff_semaphore.cu` | Extracts semaphore address/payload from QMD DW[9,15-18] and searches mapped memory for the written value. |

### Methodology

1. Launch kernel via `cudaLaunchKernel()` with known parameters
2. Call `cudaDeviceSynchronize()` to ensure QMD is populated
3. Read `/proc/self/maps` to enumerate all readable memory regions
4. Scan each region for QMD/cbuf0 signatures
5. Verify by cross-checking known values (GPU pointers, grid dims, N)
6. Dump the full structure and decode field-by-field

This "observe what CUDA does, then reproduce it" approach was essential.
The QMD v5 format is not publicly documented — the bit positions come from
tinygrad's autogenerated headers (nv_570.py) and NVIDIA's open-source QMD
class definitions (clcdc0qmd.h), but many practical details (which fields
are required, what values the driver uses, the cbuf0 template layout) can
only be discovered empirically.

## 11. Constant Buffer 3 (cbuf3) — `__constant__` Memory on SM120

Some kernels read data from `__constant__` memory. On **SM120 (Blackwell)**,
`__constant__` maps to constant buffer **3** (not 2 as on older architectures).
For example, `fft512_mel_log_kernel` reads a 504-entry mel filterbank table
from `__constant__`.

### CUBIN parsing

The `.nv.constant3` ELF section is a **global** (not per-kernel) section that
contains all `__constant__` data for the compilation unit. On SM120, there are
no `.nv.constant2` sections at all.

```
.nv.constant0.<name>  →  cbuf0_size (per-kernel: driver template + kernel args)
.nv.constant3          →  cbuf3_size (global: all __constant__ data, shared across kernels)
```

For the mel filterbank: `.nv.constant3` is 4032 bytes = 504 entries × 8 bytes,
exactly matching the `c_mel_fb[504]` array.

### QMD setup for cbuf3

Same encoding as cbuf0 (DW[42-43]) but at DW[48-49] (idx=3):

```
DW[48] = (cbuf3_gpu_addr >> 6) & 0xFFFFFFFF
DW[49] = ((cbuf3_gpu_addr >> 38) & 0x7FFFF) | ((cbuf3_size >> 4) << 19)
DW[58] |= (1 << 12)    // set cbuf3 valid bit (bit 1864 = 1856 + 3*4)
```

The cbuf3 data is uploaded to a separate GPU allocation. The kernel reads it
via `LDC c[0x3][offset]` instructions.

**Discovery**: Initially we assumed cbuf2 (matching older architectures). All
mel outputs were `log(eps) = -16.635`, indicating zero weights. `readelf -S`
on the cubin revealed no `.nv.constant2` sections — only `.nv.constant3`
(4032 bytes). Changing to cbuf3 (DW[48-49], valid bit 12) fixed it.

## 12. Dependent QMD Chaining — Unlimited Dispatches

### The problem

The GPFIFO controller on Blackwell permanently stalls after processing any
pushbuffer entry that contains a SEND_PCAS command. The original workaround
was putting ALL dispatches in ONE pushbuffer entry, but this capped out at
~16 kernels due to pushbuffer size limits. Full inference requires 100+
kernel launches per frame.

### The solution: hardware chaining

Instead of issuing SEND_PCAS for each kernel, only the FIRST kernel uses
SEND_PCAS. Each subsequent kernel is linked via `dependent_qmd0` fields in
the **previous** kernel's QMD. The GPU hardware auto-dispatches the next
kernel when the previous completes.

```
Previous QMD:
  bit 336      = DEPENDENT_QMD0_ENABLE (1)
  bits 339:337 = DEPENDENT_QMD0_ACTION (1 = QMD_SCHEDULE)
  bit 340      = DEPENDENT_QMD0_PREFETCH (1)
  bits 415:384 = DEPENDENT_QMD0_POINTER (next_qmd_gpu_addr >> 8)
```

QMD bit positions are from NVIDIA's `clc5c0qmd.h` (open-gpu-doc) and
tinygrad's QMD field definitions.

### Implementation

```
First kernel:
  SEND_PCAS_A ← qmd0 address
  SEND_SIGNALING_PCAS2_B ← 9

Second kernel:
  Write dependent_qmd0 fields into qmd0 → qmd1 address
  (no SEND_PCAS needed)

Third kernel:
  Write dependent_qmd0 fields into qmd1 → qmd2 address
  ...

Last kernel:
  Set QMD release semaphore on last QMD
  Submit ONE GPFIFO entry
```

This gives unlimited sequential dispatches with a single GPFIFO submission.

### Critical: clearing DEPENDENT_QMD0_ENABLE on the last kernel

The CUDA-captured QMD template had `DW[10] = 0x00190000` which included
`DEPENDENT_QMD0_ENABLE=1` (bit 336) with a garbage pointer in `DW[12]`.
Before chaining, all kernels were dispatched via SEND_PCAS, and the GPU
ignores `dependent_qmd0` fields for SEND_PCAS-dispatched kernels — so
the stale bits were harmless.

With chaining, kernels 2–N are dispatched **as dependents**, and the GPU
DOES check their `dependent_qmd0` fields on completion. The chaining code
correctly overwrites these fields on kernels 1 through N-1 (to point to
the next kernel). But the **last kernel** kept the template's `ENABLE=1`
with garbage pointer `0x02070100`, causing the GPU to chase
`0x02070100 << 8 = 0x20701000000` — unmapped memory → GPU fault →
semaphore never written → infinite CPU poll.

**Fix**: Clear `dependent_qmd0` fields from the template (`DW[10]=0`,
`DW[12]=0`) and explicitly disable on the last QMD in `wait_kernel()`:
```cpp
last->set(336, 336, 0);  // Disable dependent_qmd0 on last kernel
```

### QMD release semaphore

Instead of the old HOST SEM approach (which required a separate GPFIFO
entry), the last kernel's QMD is configured with a release semaphore that
fires on kernel completion:

```
QMD release semaphore fields:
  bit 288          = RELEASE0_ENABLE (1)
  bits 290:289     = RELEASE_STRUCTURE_SIZE (1 = 64-bit)
  bit 300          = RELEASE0_REDUCTION_ENABLE (0)
  bits 536:480     = RELEASE0_ADDRESS (byte address)
  bits 607:544     = RELEASE0_PAYLOAD (counter value)
```

A HOST SEM with `RELEASE_WFI` (bit 20 of SEM_EXECUTE) is also emitted as
backup, ensuring all in-flight work completes before the semaphore fires.

## 13. CUTLASS GEMM via Cudaless Path

The inference pipeline uses 10 CUTLASS GEMM kernel instantiations for all
matrix multiplies. Running these via the cudaless path required solving three
sub-problems: compiling CUTLASS to cubin, constructing Params structs, and
identifying kernels in the cubin.

### 13a. Compiling CUTLASS to cubin

```bash
nvcc -std=c++17 -O3 --cubin -arch=sm_120 \
  -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include \
  src/cutlass_gemm.cu -o cutlass_gemm.cubin
```

Produces a 502KB cubin with 10 kernels. All use SM80 TensorOp (mma.sync
16x8x16) which runs on SM120 via forward compatibility.

### 13b. The Params construction problem

CUTLASS kernels take a `Params` struct (368 bytes) in cbuf0, not raw
pointers. The Params struct contains computed fields (iterator params,
grid tiling info, swizzle log) that depend on the CUTLASS template
instantiation. We can't construct these without the CUTLASS template
machinery.

**Solution: nvcc-compiled bridge**

`src/cutlass_params.cu` is compiled with nvcc to instantiate all 10 CUTLASS
templates and expose `extern "C"` functions:

```c
// Returns params_size, fills raw bytes + grid/block/smem
int cutlass_params_nn_64x64_64_s6(
    void* out_params, int out_sz,
    int* grid, int* block, int* smem,
    int M, int N, int K,
    uint64_t A, int ldA, uint64_t B, int ldB,
    uint64_t C, int ldC,
    uint16_t alpha_fp16, uint16_t beta_fp16);
```

The bridge calls `GemmOp::initialize()` (pure host-side, no CUDA runtime
calls) to fill the Params, then `memcpy`s the raw bytes out.

**Eliminating libcudart dependency**: nvcc generates CUDA runtime
registration symbols (`__cudaRegisterFatBinary`, etc.) even for
host-only `.cu` files. We provide empty stubs in `src/cuda_stubs.cpp`
to satisfy the linker without pulling in libcudart:

```cpp
extern "C" {
    void** __cudaRegisterFatBinary(void*) { static void* v = 0; return &v; }
    void __cudaRegisterFatBinaryEnd(void**) {}
    void __cudaUnregisterFatBinary(void**) {}
    void __cudaRegisterVar(void**, char*, char*, const char*,
                           int, size_t, int, int) {}
}
```

### 13c. CUTLASS Params layout (reverse-engineered)

Using `tools/sniff_cutlass2.cu` (constructs Params via CUTLASS API and dumps
raw bytes), we verified the 368-byte Params layout for all variants:

**Non-batched Gemm Params (368 bytes):**
```
[  0] problem_size.m (int32)      [  4] problem_size.n (int32)
[  8] problem_size.k (int32)      [ 12] grid_tiled_shape.m (int32)
[ 16] grid_tiled_shape.n (int32)  [ 20] grid_tiled_shape.k (int32)
[ 24] swizzle_log_tile (int32)    [ 28] params_A (iterator params)
...
[ 64] ref_A.data (ptr64)          [ 72] ref_A.stride (int32)
...
[112] ref_B.data (ptr64)          [120] ref_B.stride (int32)
...
[192] ref_C.data (ptr64)          [200] ref_C.stride (int32)
...
[272] ref_D.data (ptr64)          [280] ref_D.stride (int32)
[288] output_op.alpha (fp16)      [290] output_op.beta (fp16)
[296] semaphore (ptr64)           [304] gemm_k_size (int32)
```

**Batched GemmBatched Params** add stride fields at offsets 80, 136, 224
and `batch_count` at offset 360.

### 13d. Kernel identification in cubin

CUTLASS kernel names are 2000+ character mangled C++ symbols. The cubin
loader identifies each of the 10 variants by analyzing mangled name features:

| Feature | Detection |
|---------|-----------|
| Batched vs non-batched | `"GemmBatched"` substring |
| Tile K=64 vs K=32 | `"64ELi64ELi64E"` vs `"64ELi64ELi32E"` |
| Tile 128x128 | `"128ELi128E"` |
| Stage count | `"EELi3ELN"`, `"EELi6ELN"`, `"EELi10ELN"` |
| Pipelined (stage=2) | `"Pipelined"` |
| Alignment=2 epilogue | `"ISD_Li2ELb0EEELb0E"` |
| Epilogue alignment=1 | `"LinearCombinationISD_Li1E"` |
| NN vs TN | Count of `"Congruous"` (0=NN for k=64, >0=TN for k=64) |

All 10 variants are uniquely identified by these features.

### 13e. Row-major to column-major conversion

The user-facing API uses row-major convention (matching PyTorch/ONNX), but
CUTLASS uses column-major internally. The conversion is:

```
Row-major: Y[m,n] = X[m,k] @ W[k,n]
Col-major: Y'[n,m] = W'[n,k] @ X'[k,m]    (just swapping A/B and m/n)

CUTLASS args: problem=(n, m, k), A=W(ldA=n), B=X(ldB=k), C=Y(ldC=n)
```

For NT (transposed W): `A=W^T(ldA=k)` instead.

### 13f. The 10 kernel variants

| # | Variant | Tile | K | Stages | Epilogue | Use |
|---|---------|------|---|--------|----------|-----|
| 1 | GemmNN_64x64_64_s6 | 64×64 | 64 | 6 | align=8 | General NN matmul |
| 2 | GemmNN_64x64_32_s10 | 64×64 | 32 | 10 | align=8 | Small-K NN |
| 3 | GemmNN_64x64_32_s3 | 64×64 | 32 | 3 | align=8 | Low-latency NN |
| 4 | GemmNN_64x64_32_s6_a2 | 64×64 | 32 | 6 | align=2 | Narrow output NN |
| 5 | GemmNN_128x128_32_s5 | 128×128 | 32 | 5 | align=8 | Large NN |
| 6 | GemmTN_64x64_64_s6 | 64×64 | 64 | 6 | align=8 | Transposed weight |
| 7 | BatchNN_64x64_64_s6 | 64×64 | 64 | 6 | align=8 | Multi-head attention |
| 8 | BatchNN_64x64_32_s2 | 64×64 | 32 | 2 | align=8 | Small batch NN |
| 9 | BatchTN_64x64_64_s6 | 64×64 | 64 | 6 | align=8 | Batched transposed |
| 10 | BatchTN_64x64_64_s6_e1 | 64×64 | 64 | 6 | align=1 | Narrow batched TN |

## 14. Files

```
src/gpu.h                — single-header GPU driver (~1050 lines)
                           init, alloc, launch_kernel (dependent QMD chaining),
                           wait_kernel (QMD release semaphore), prepare_cbuf0
src/cubin_loader.h       — CUBIN ELF parser (~350 lines)
                           .text, .nv.info, .nv.constant0, .nv.constant3, .nv.shared
src/cutlass_cudaless.h   — CUTLASS GEMM wrapper (~250 lines)
                           cubin load, kernel identification, row→col conversion
src/cutlass_params.cu    — nvcc-compiled Params bridge (~340 lines)
                           10 extern "C" functions, one per GEMM variant
src/cuda_stubs.cpp       — CUDA runtime stubs (~20 lines)
                           eliminates libcudart dependency
tests/test_launch.cpp    — single-kernel end-to-end test
tests/test_kernels.cpp   — 24-kernel test harness (all custom kernels + fft512_mel_log)
tests/test_cutlass_cudaless.cpp — CUTLASS GEMM test (4 variants vs CPU reference)
tools/sniff_cutlass2.cu  — RE tool: dump CUTLASS Params layout
tools/sniff_cbuf0.cu     — RE tool: dump cbuf0 from CUDA
tools/sniff_qmd2.cu      — RE tool: dump QMD from CUDA
tools/sniff_semaphore.cu — RE tool: find semaphore writes
third_party/nv-headers/  — vendored NVIDIA headers (MIT)
```

## 15. Build

```bash
# Compile custom kernels to cubin
make kernels.cubin

# Compile CUTLASS kernels to cubin
make cutlass_gemm.cubin

# Build custom kernel tests (zero NVIDIA runtime deps)
make test_kernels

# Build CUTLASS GEMM test (zero libcudart deps — uses stubs)
make test_cutlass_cudaless

# Verify: no CUDA libs
ldd test_cutlass_cudaless
# → linux-vdso, libstdc++, libm, libgcc_s, libc, ld-linux
```

## 16. CUTLASS GEMM Debugging — Open Issue

### Status: PARAMS ARE WRONG (not a launch/QMD/cbuf0 issue)

All 4 CUTLASS GEMM tests fail via cudaless. **But they also fail when launched
via the CUDA driver API (`cuLaunchKernel`)** with the exact same params. This
definitively proves the issue is in the params construction, not in QMD, cbuf0,
or any part of the cudaless launch path.

### Proof: test_cutlass_cuda.cu

`tests/test_cutlass_cuda.cu` loads `cutlass_gemm.cubin` via `cuModuleLoad`,
builds params using the same `cutlass_params_nn_64x64_64_s6()` bridge, and
launches via `cuLaunchKernel`. Result:

```
CUDA DRIVER API: FAIL (4070/4096 errors, max_rel=2413.6792) — PARAMS ARE WRONG
```

Exact same error count and max_rel as the cudaless path. The GPU output values
are identical between both paths. This rules out every cudaless-specific theory.

### Failure pattern

| Test | Result | Notes |
|------|--------|-------|
| Zero inputs (A=0, B=0) | PASS | Output is all zeros |
| All-ones (A=1, B=1) | PASS | Output is all 64.0 (= K) |
| Identity (A=I, B=sequential) | PASS | Output equals B |
| Random × random (64×64×64) | **FAIL** | 4070/4096 wrong, max_rel=2413 |
| Random gemm_nt (64×128×64) | **FAIL** | 8131/8192 wrong |
| Batched NN (4×64×64×64) | **FAIL** | timeout + wrong |
| Batched NT (4×64×64×64) | **FAIL** | timeout + all zeros |

The identity test passing is **misleading** — it doesn't prove correctness
because `I @ B = B` regardless of whether A and B are swapped, transposed,
or read with wrong strides. Similarly, uniform inputs (zeros, ones) pass
because wrong addressing yields the same values.

### Root cause analysis: the ColumnMajor double-swap

The CUTLASS `Gemm` class has a **partial specialization for ColumnMajor output**
(`cutlass/gemm/device/gemm.h` line 572). When all three matrices are ColumnMajor
(as in our NN types), this specialization activates and does a transparent
swap in `to_underlying_arguments()` (line 700):

```cpp
static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},  // swap M,N
      {args.ref_B.data(), args.ref_B.stride(0)},   // swap A←B
      {args.ref_A.data(), args.ref_A.stride(0)},   // swap B←A
      ...
    );
}
```

Our `cutlass_cudaless.h` already does a manual row→col swap:
```cpp
// gemm_nn: Row→Col conversion
launch_gemm(..., n, m, k, W, n, X, k, Y, n, ...)  // swap M↔N, A↔B
```

So the chain is:
1. **Our code** swaps: M=n, N=m, A=W(B_row), B=X(A_row)
2. **CUTLASS ColumnMajor specialization** swaps back: M=m, N=n, A=X(A_row), B=W(B_row)
3. **Underlying RowMajor kernel** computes: C = X @ W = A_row @ B_row ✓

The double-swap cancels out and the math is correct. **But the params struct
is built by the underlying RowMajor operator**, and the SM120 cubin kernel
was compiled with the **ColumnMajor types** (which use the RowMajor underlying
kernel). The params layout should match since `GemmKernel` is delegated through.

### What we verified is NOT the problem

| Theory | Status | How eliminated |
|--------|--------|----------------|
| QMD fields wrong | ❌ | Same failure via cuLaunchKernel (CUDA sets QMD) |
| cbuf0[0x358] descriptor | ❌ | Same failure via cuLaunchKernel |
| Shared memory config | ❌ | Same failure via cuLaunchKernel |
| Memory allocation/coherence | ❌ | Input readback matches, same failure via CUDA allocs |
| Row→col conversion math | ❌ | Verified algebraically, identity test passes |
| params_ at wrong offset in GemmOp | ❌ | Verified: params_ is at offset 0, no vtable, no base class |
| cudaless launch path | ❌ | **Definitively eliminated by test_cutlass_cuda.cu** |

### What IS likely wrong

The params are constructed using **SM80 CUTLASS types** (`cutlass::arch::Sm80`)
but the cubin is compiled for **SM120**. The `cutlass_params.cu` bridge compiles
with `-arch=sm_80` while `cutlass_gemm.cubin` compiles with `-arch=sm_120`.

Potential mismatch: the SM80 `Params` struct layout may differ from what the
SM120-compiled kernel expects. The SM80 types use `Sm80` arch tag, `mma.sync`
instruction shapes, and SM80 iterator policies. When compiled to SM120 SASS,
nvcc may:
- Reorder struct fields for SM120 alignment
- Use different iterator parameters
- Compute different tiling constants

**This is the #1 hypothesis to investigate next.**

### Concrete next steps

1. **Compile cutlass_params.cu with `-arch=sm_120`** instead of sm_80, to
   ensure params struct layout matches the cubin. This requires either:
   - Making it a device compilation (cubin) and extracting params differently
   - Using `-arch=sm_120 --device-c` to get host code with SM120 types

2. **Dump params from a working CUDA CUTLASS launch** using `tools/sniff_cutlass2.cu`
   and compare byte-for-byte against our bridge output

3. **Try the existing CUDA CUTLASS path** (`src/cutlass_gemm.cu` compiled with
   `-arch=sm_80`) to verify it actually works — this uses the same SM80 types
   but launches via `cudaLaunchKernel` which may handle the params differently

### Test commands

```bash
# Build and run CUDA driver API test (proves params are wrong)
nvcc -std=c++17 -O3 -arch=sm_120 -Isrc -Ithird_party/cutlass/include \
  -Ithird_party/cutlass/tools/util/include --expt-relaxed-constexpr \
  tests/test_cutlass_cuda.cu src/cutlass_params.o -lcuda -lcudart \
  -o test_cutlass_cuda && ./test_cutlass_cuda

# Build and run cudaless test
make test_cutlass_cudaless && ./test_cutlass_cudaless
```

## 17. What's Next

- **Fix CUTLASS params** — investigate SM80 vs SM120 params layout mismatch
- **Full inference**: Wire up paraketto_cudaless.cpp end-to-end
- **Performance**: Measure RTFx, compare against CUDA path

## 18. Key Numbers

| Metric | Value |
|--------|-------|
| Full test (init + 24 kernels + verify) | 150ms wall clock |
| GPU init time | ~50ms (20 RM ioctls + UVM setup) |
| Kernel launch overhead | <10us (QMD build + GPFIFO submit) |
| Semaphore latency | ~140 spins for 4096-element FP16 kernel |
| Custom kernels tested | 24/24 passing (23 original + fft512_mel_log via cbuf3) |
| CUTLASS variants | 10 (6 non-batched, 4 batched) |
| CUTLASS cubin size | 502 KB (sm_120) |
| Params struct size | 368 bytes (all variants) |
| Total NVIDIA userspace code | 0 bytes |
| Headers vendored | ~50 files from open-gpu-kernel-modules |
| Binary deps | libc, libstdc++, libm only |
