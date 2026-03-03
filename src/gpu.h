// gpu.h — Direct GPU access via ioctls, bypassing libcuda.so
//
// Uses NVIDIA open-gpu-kernel-modules headers (MIT licensed) for all
// struct definitions and constants. No more hand-guessing flag values.
//
// Reference: tinygrad ops_nv.py for the init sequence.

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// =========================================================================
// NVIDIA SDK headers (from open-gpu-kernel-modules, MIT license)
// =========================================================================

#include "nvos.h"           // NVOS21, NVOS33, NVOS46, NVOS54, alloc params, flags
#include "nv-ioctl.h"       // nv_ioctl_card_info_t, nv_ioctl_register_fd_t
#include "nv_escape.h"      // NV_ESC_RM_ALLOC, NV_ESC_RM_CONTROL, etc.
#include "nv-ioctl-numbers.h" // NV_ESC_CARD_INFO, NV_ESC_REGISTER_FD

// Class headers
#include "class/cl0000.h"   // NV01_ROOT
#include "class/cl0040.h"   // NV1_MEMORY_USER
#include "class/cl0070.h"   // NV01_MEMORY_VIRTUAL
#include "class/cl0080.h"   // NV01_DEVICE_0
#include "class/cl2080.h"   // NV20_SUBDEVICE_0
#include "class/cl2080_notification.h" // NV2080_ENGINE_TYPE_GRAPHICS
#include "class/cl83de.h"   // GT200_DEBUGGER
#include "class/cl9067.h"   // FERMI_CONTEXT_SHARE_A
#include "class/cl90f1.h"   // FERMI_VASPACE_A
#include "class/cla06c.h"   // KEPLER_CHANNEL_GROUP_A
#include "class/clc661.h"   // HOPPER_USERMODE_A
#include "class/clc96f.h"   // BLACKWELL_CHANNEL_GPFIFO_A
#include "class/clcab5.h"   // BLACKWELL_DMA_COPY_B
#include "class/clcec0.h"   // BLACKWELL_COMPUTE_B

// Control command headers
#include "ctrl/ctrl0000/ctrl0000gpu.h"  // NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2
#include "ctrl/ctrl2080/ctrl2080gpu.h"  // NV2080_CTRL_CMD_GPU_GET_GID_INFO
#include "ctrl/ctrl2080/ctrl2080perf.h" // NV2080_CTRL_CMD_PERF_BOOST
#include "ctrl/ctrlc36f.h"             // NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN
#include "ctrl/ctrla06c.h"             // NVA06C_CTRL_CMD_GPFIFO_SCHEDULE

// UVM headers — on Linux, UVM_IOCTL_BASE(i) = i (plain ioctl number)
#include "uvm_linux_ioctl.h"

// =========================================================================
// NVOS33 with fd (Linux-specific, not in upstream headers)
// =========================================================================

struct NVOS33_with_fd {
    NVOS33_PARAMETERS params;
    NvS32 fd;
    NvU32 _pad;
};

// =========================================================================
// QMD v5 helper (Blackwell, 96 DWORDs = 384 bytes)
// =========================================================================

struct QMD {
    uint32_t dw[96];

    void set(int hi_bit, int lo_bit, uint32_t value) {
        int dw_idx = lo_bit / 32;
        int lo = lo_bit % 32;
        int hi = hi_bit % 32;
        uint32_t mask = ((2u << (hi - lo)) - 1) << lo;
        dw[dw_idx] = (dw[dw_idx] & ~mask) | ((value << lo) & mask);
    }

    uint32_t get(int hi_bit, int lo_bit) const {
        int dw_idx = lo_bit / 32;
        int lo = lo_bit % 32;
        int hi = hi_bit % 32;
        uint32_t mask = ((2u << (hi - lo)) - 1) << lo;
        return (dw[dw_idx] & mask) >> lo;
    }

    // Field setters (bit positions from clcdc0qmd.h / tinygrad nv_570.py)
    void set_qmd_major_version(uint32_t v)         { set(471, 468, v); }
    void set_qmd_type(uint32_t v)                  { set(153, 151, v); }
    void set_qmd_group_id(uint32_t v)              { set(149, 144, v); }
    void set_api_visible_call_limit(uint32_t v)    { set(456, 456, v); }
    void set_sampler_index(uint32_t v)             { set(457, 457, v); }
    void set_sass_version(uint32_t v)              { set(455, 448, v); }
    void set_cwd_membar_type(uint32_t v)           { set(625, 624, v); }
    void set_barrier_count(uint32_t v)             { set(1141, 1137, v); }

    void set_program_address(uint64_t addr) {
        uint64_t shifted = addr >> 4;
        set(1055, 1024, (uint32_t)(shifted & 0xFFFFFFFF));
        set(1076, 1056, (uint32_t)(shifted >> 32));
    }

    void set_program_prefetch(uint64_t addr, uint32_t size_bytes) {
        set(1085, 1077, (size_bytes >> 8) < 0x1FF ? (size_bytes >> 8) : 0x1FF);
        set(1087, 1086, 1); // PREFETCH_TYPE = CTA
        set(1919, 1888, (uint32_t)(addr >> 8));
        set(1936, 1920, (uint32_t)(addr >> 40));
    }

    void set_register_count(uint32_t v)            { set(1136, 1128, v); }

    void set_shared_memory(uint32_t bytes) {
        set(1162, 1152, bytes >> 7);
        // smem_cfg = min(tier_kb*1024 for tiers in [32,64,100] if >= bytes) / 4096 + 1
        // Per tinygrad: 32KB→9, 64KB→17, 100KB→26
        uint32_t smem_cfg;
        if (bytes <= 32 * 1024) smem_cfg = 9;
        else if (bytes <= 64 * 1024) smem_cfg = 17;
        else smem_cfg = 26;
        set(1168, 1163, smem_cfg);  // min
        set(1180, 1175, smem_cfg);  // target
        set(1174, 1169, 0x1A);     // max (always 100KB encoded)
    }

    void set_local_memory(uint32_t low_bytes, uint32_t high_bytes) {
        set(1199, 1184, low_bytes >> 4);
        set(1215, 1200, high_bytes >> 4);
    }

    void set_grid(uint32_t x, uint32_t y, uint32_t z) {
        set(1279, 1248, x);
        set(1295, 1280, y);
        set(1327, 1312, z);
    }

    void set_block(uint32_t x, uint32_t y, uint32_t z) {
        set(1103, 1088, x);
        set(1119, 1104, y);
        set(1127, 1120, z);
    }

    void set_constant_buffer(int idx, uint64_t addr, uint32_t size_bytes) {
        uint64_t shifted_addr = addr >> 6;
        int base = 1344 + idx * 64;
        set(base + 31, base, (uint32_t)(shifted_addr & 0xFFFFFFFF));
        set(base + 50, base + 32, (uint32_t)(shifted_addr >> 32));
        set(base + 63, base + 51, size_bytes >> 4);
        set(1856 + idx * 4, 1856 + idx * 4, 1);      // valid
        set(1859 + idx * 4, 1859 + idx * 4, 1);      // invalidate
    }

    // Authoritative QMD v5 semaphore bit positions (from tinygrad nv_570.py autogen)
    void set_release_semaphore(int idx, uint64_t addr, uint64_t payload) {
        // RELEASE0_ENABLE(288), RELEASE_STRUCTURE_SIZE(290:289)=0 (FOUR_WORDS),
        // RELEASE_PAYLOAD64B(300)
        set(288, 288, 1);                              // RELEASE0_ENABLE
        set(290, 289, 0);                              // RELEASE_STRUCTURE_SIZE = FOUR_WORDS
        set(300, 300, 1);                              // RELEASE_PAYLOAD64B
        // Semaphore 0 address: bits 511:480 = lower, bits 536:512 = upper
        set(511, 480, (uint32_t)(addr & 0xFFFFFFFF));  // RELEASE0_ADDRESS_LOWER
        set(536, 512, (uint32_t)(addr >> 32));         // RELEASE0_ADDRESS_UPPER
        // Semaphore 0 payload: bits 575:544 = lower, bits 607:576 = upper
        set(575, 544, (uint32_t)(payload & 0xFFFFFFFF)); // RELEASE0_PAYLOAD_LOWER
        set(607, 576, (uint32_t)(payload >> 32));        // RELEASE0_PAYLOAD_UPPER
    }

    void set_invalidate_caches() {
        set(472, 472, 1); // texture header
        set(473, 473, 1); // texture sampler
        set(474, 474, 1); // texture data
        set(475, 475, 1); // shader data
        set(477, 477, 1); // shader constant
    }
};

// =========================================================================
// GPU class — direct ioctl driver
// =========================================================================

class GPU {
public:
    // Device file descriptors
    int fd_ctl = -1;
    int fd_dev = -1;
    int fd_uvm = -1;
    int fd_uvm2 = -1;

    // RM handles
    NvHandle h_root = 0;
    NvHandle h_device = 0;
    NvHandle h_subdevice = 0;
    NvHandle h_virtmem = 0;
    NvHandle h_vaspace = 0;
    NvHandle h_usermode = 0;
    NvHandle h_channel_group = 0;
    NvHandle h_ctxshare = 0;
    NvHandle h_gpfifo_mem = 0;
    NvHandle h_compute_channel = 0;
    NvHandle h_compute_obj = 0;
    NvHandle h_copy_channel = 0;
    NvHandle h_copy_obj = 0;

    // GPU info
    NvU32 gpu_id = 0;
    NvU32 minor_number = 0;
    NvProcessorUuid gpu_uuid = {};

    // MMIO doorbell
    volatile uint32_t* gpu_mmio = nullptr;

    // GPFIFO
    void*     gpfifo_cpu = nullptr;
    uint64_t  gpfifo_gpu = 0;
    uint32_t  compute_token = 0;
    uint32_t  copy_token = 0;
    uint32_t  compute_put = 0;
    uint32_t  copy_put = 0;

    // Command queue
    void*     cmdq_cpu = nullptr;
    uint64_t  cmdq_gpu = 0;
    uint32_t  cmdq_offset = 0;

    // VA allocator (simple bump)
    uint64_t  va_next = 0x1000000000ULL;  // tinygrad: low range starts at 0x1000000000

    // Handle counter
    NvHandle  next_handle = 0xCAFE0000;

    // =========================================================================
    // ioctl wrappers
    // =========================================================================

    int nv_ioctl(int fd, uint32_t nr, void* data, size_t size) {
        unsigned long cmd = (3u << 30) | ((size & 0x1FFF) << 16) | (0x46 << 8) | (nr & 0xFF);
        return ::ioctl(fd, cmd, data);
    }

    int uvm_ioctl(uint32_t cmd, void* data) {
        return ::ioctl(fd_uvm, cmd, data);
    }

    NvHandle new_handle() { return next_handle++; }

    NvHandle rm_alloc(NvHandle parent, NvU32 cls,
                      void* params = nullptr, NvU32 params_size = 0) {
        NVOS21_PARAMETERS args = {};
        args.hRoot = h_root;
        args.hObjectParent = parent;
        args.hObjectNew = new_handle();
        args.hClass = cls;
        args.pAllocParms = params;
        args.paramsSize = params_size;

        int ret = nv_ioctl(fd_ctl, NV_ESC_RM_ALLOC, &args, sizeof(args));
        if (ret != 0 || args.status != 0) {
            fprintf(stderr, "rm_alloc(parent=0x%x, class=0x%x) failed: ret=%d status=0x%x\n",
                    parent, cls, ret, args.status);
            return 0;
        }
        return args.hObjectNew;
    }

    NvHandle rm_alloc_root() {
        NVOS21_PARAMETERS args = {};
        args.hRoot = 0;
        args.hObjectParent = 0;
        args.hObjectNew = new_handle();
        args.hClass = NV01_ROOT_CLIENT;
        args.pAllocParms = nullptr;
        args.paramsSize = 0;

        int ret = nv_ioctl(fd_ctl, NV_ESC_RM_ALLOC, &args, sizeof(args));
        if (ret != 0 || args.status != 0) {
            fprintf(stderr, "rm_alloc_root failed: ret=%d status=0x%x\n", ret, args.status);
            return 0;
        }
        return args.hObjectNew;
    }

    int rm_control(NvHandle object, NvU32 cmd,
                   void* params = nullptr, NvU32 params_size = 0) {
        NVOS54_PARAMETERS args = {};
        args.hClient = h_root;
        args.hObject = object;
        args.cmd = cmd;
        args.flags = 0;
        args.params = params;
        args.paramsSize = params_size;

        int ret = nv_ioctl(fd_ctl, NV_ESC_RM_CONTROL, &args, sizeof(args));
        if (ret != 0 || args.status != 0) {
            fprintf(stderr, "rm_control(obj=0x%x, cmd=0x%x) failed: ret=%d status=0x%x\n",
                    object, cmd, ret, args.status);
            return -1;
        }
        return 0;
    }

    // =========================================================================
    // VA allocator
    // =========================================================================

    uint64_t alloc_va(uint64_t size, uint64_t align = 0x1000) {
        va_next = (va_next + align - 1) & ~(align - 1);
        uint64_t addr = va_next;
        va_next += size;
        return addr;
    }

    // =========================================================================
    // CPU mapping (tinygrad pattern: fresh fd + RM map + mmap)
    // =========================================================================

    void* map_to_cpu(NvHandle h_parent, NvHandle h_mem, uint64_t size,
                     NvU32 flags = 0) {
        char dev_path[32];
        snprintf(dev_path, sizeof(dev_path), "/dev/nvidia%d", minor_number);
        int map_fd = open(dev_path, O_RDWR | O_CLOEXEC);
        if (map_fd < 0) return nullptr;

        nv_ioctl_register_fd_t reg = {};
        reg.ctl_fd = fd_ctl;
        nv_ioctl(map_fd, NV_ESC_REGISTER_FD, &reg, sizeof(reg));

        NVOS33_with_fd map = {};
        map.fd = map_fd;
        map.params.hClient = h_root;
        map.params.hDevice = h_parent;
        map.params.hMemory = h_mem;
        map.params.length = size;
        map.params.flags = flags;
        int ret = nv_ioctl(fd_ctl, NV_ESC_RM_MAP_MEMORY, &map, sizeof(map));
        if (ret != 0 || map.params.status != 0) {
            fprintf(stderr, "map_to_cpu: failed ret=%d status=0x%x\n", ret, map.params.status);
            close(map_fd);
            return nullptr;
        }

        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, map_fd, 0);
        if (ptr == MAP_FAILED) {
            perror("map_to_cpu: mmap");
            close(map_fd);
            return nullptr;
        }
        return ptr;
    }

    // =========================================================================
    // Initialization (follows tinygrad ops_nv.py sequence exactly)
    // =========================================================================

    bool init() {
        // 1. Open device files
        fd_ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
        if (fd_ctl < 0) { perror("open /dev/nvidiactl"); return false; }

        fd_uvm = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        if (fd_uvm < 0) { perror("open /dev/nvidia-uvm"); return false; }

        fd_uvm2 = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        if (fd_uvm2 < 0) { perror("open /dev/nvidia-uvm (2nd)"); return false; }

        // 2. Create root RM client
        h_root = rm_alloc_root();
        if (!h_root) return false;

        // 3. Initialize UVM
        UVM_INITIALIZE_PARAMS uvm_init = {};
        if (uvm_ioctl((UVM_INITIALIZE), &uvm_init) != 0 || uvm_init.rmStatus != 0) {
            fprintf(stderr, "UVM_INITIALIZE failed: %d\n", uvm_init.rmStatus);
            return false;
        }

        UVM_MM_INITIALIZE_PARAMS mm_init = {};
        mm_init.uvmFd = fd_uvm;
        ::ioctl(fd_uvm2, (UVM_MM_INITIALIZE), &mm_init);

        // 4. Enumerate GPUs
        nv_ioctl_card_info_t cards[64] = {};
        nv_ioctl(fd_ctl, NV_ESC_CARD_INFO, cards, sizeof(cards));

        for (int i = 0; i < 64; i++) {
            if (cards[i].valid) {
                gpu_id = cards[i].gpu_id;
                minor_number = cards[i].minor_number;
                fprintf(stderr, "GPU %d: gpu_id=0x%x, minor=%d\n", i, gpu_id, minor_number);
                break;
            }
        }
        if (gpu_id == 0) { fprintf(stderr, "No GPU found\n"); return false; }

        // 5. Open GPU device file
        char dev_path[32];
        snprintf(dev_path, sizeof(dev_path), "/dev/nvidia%d", minor_number);
        fd_dev = open(dev_path, O_RDWR | O_CLOEXEC);
        if (fd_dev < 0) { perror(dev_path); return false; }

        nv_ioctl_register_fd_t reg_fd = {};
        reg_fd.ctl_fd = fd_ctl;
        nv_ioctl(fd_dev, NV_ESC_REGISTER_FD, &reg_fd, sizeof(reg_fd));

        // 6. Create device (tinygrad queries deviceInstance, default 0 for single GPU)
        NV0080_ALLOC_PARAMETERS dev_params = {};
        dev_params.deviceId = 0;
        dev_params.hClientShare = h_root;
        dev_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
        h_device = rm_alloc(h_root, NV01_DEVICE_0, &dev_params, sizeof(dev_params));
        if (!h_device) return false;

        // 7. Create subdevice
        NV2080_ALLOC_PARAMETERS subdev_params = {};
        h_subdevice = rm_alloc(h_device, NV20_SUBDEVICE_0, &subdev_params, sizeof(subdev_params));
        if (!h_subdevice) return false;

        // 8. Allocate virtual memory handle
        NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS vmem = {};
        vmem.limit = 0x1FFFFFFFFFFFF;
        h_virtmem = rm_alloc(h_device, NV01_MEMORY_VIRTUAL, &vmem, sizeof(vmem));
        if (!h_virtmem) return false;

        // 9. Setup usermode MMIO for doorbell
        h_usermode = rm_alloc(h_subdevice, HOPPER_USERMODE_A);
        if (!h_usermode) return false;

        gpu_mmio = (volatile uint32_t*)map_to_cpu(h_subdevice, h_usermode, 0x10000);
        if (!gpu_mmio) {
            fprintf(stderr, "Failed to map usermode MMIO\n");
            return false;
        }

        // 10. Boost GPU to max clocks
        NV2080_CTRL_PERF_BOOST_PARAMS perf = {};
        perf.duration = 0xFFFFFFFF;
        perf.flags = (NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4)
                   | (NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6)
                   | NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX;
        rm_control(h_subdevice, NV2080_CTRL_CMD_PERF_BOOST, &perf, sizeof(perf));
        // non-fatal if fails

        // 11. Get GPU UUID
        NV2080_CTRL_GPU_GET_GID_INFO_PARAMS gid = {};
        gid.flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY;
        if (rm_control(h_subdevice, NV2080_CTRL_CMD_GPU_GET_GID_INFO,
                       &gid, sizeof(gid)) == 0) {
            memcpy(&gpu_uuid, gid.data, sizeof(gpu_uuid));
            fprintf(stderr, "GPU UUID: ");
            for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", gpu_uuid.uuid[i]);
            fprintf(stderr, "\n");
        }

        // 12. Create VA space
        fprintf(stderr, "[12] Creating VA space...\n");
        NV_VASPACE_ALLOCATION_PARAMETERS va_params = {};
        va_params.vaBase = 0x1000;
        va_params.vaSize = 0x1FFFFFB000000;
        va_params.flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING
                        | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
        h_vaspace = rm_alloc(h_device, FERMI_VASPACE_A, &va_params, sizeof(va_params));
        if (!h_vaspace) return false;
        fprintf(stderr, "[12] VA space OK: h_vaspace=0x%x\n", h_vaspace);

        // 13. Register GPU + VA space with UVM (tinygrad uses rmCtrlFd=-1 for REGISTER_GPU)
        fprintf(stderr, "[13] Registering GPU with UVM (ioctl=0x%x)...\n", (unsigned)(UVM_REGISTER_GPU));
        UVM_REGISTER_GPU_PARAMS reg_gpu = {};
        reg_gpu.gpu_uuid = gpu_uuid;
        reg_gpu.rmCtrlFd = -1;
        reg_gpu.hClient = h_root;
        if (uvm_ioctl((UVM_REGISTER_GPU), &reg_gpu) != 0)
            fprintf(stderr, "[13] UVM_REGISTER_GPU ioctl failed: errno=%d (%s)\n", errno, strerror(errno));
        else
            fprintf(stderr, "[13] UVM_REGISTER_GPU: rmStatus=%d\n", reg_gpu.rmStatus);

        UVM_REGISTER_GPU_VASPACE_PARAMS reg_va = {};
        reg_va.gpuUuid = gpu_uuid;
        reg_va.rmCtrlFd = fd_ctl;
        reg_va.hClient = h_root;
        reg_va.hVaSpace = h_vaspace;
        if (uvm_ioctl((UVM_REGISTER_GPU_VASPACE), &reg_va) != 0)
            fprintf(stderr, "[13] UVM_REGISTER_GPU_VASPACE ioctl failed\n");
        else
            fprintf(stderr, "[13] UVM_REGISTER_GPU_VASPACE: rmStatus=%d\n", reg_va.rmStatus);

        // 14. Create channel group (TSG)
        fprintf(stderr, "[14] Creating channel group...\n");
        NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS tsg = {};
        tsg.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
        h_channel_group = rm_alloc(h_device, KEPLER_CHANNEL_GROUP_A, &tsg, sizeof(tsg));
        if (!h_channel_group) return false;
        fprintf(stderr, "[14] Channel group OK: h=0x%x\n", h_channel_group);

        // 15. Allocate GPFIFO area (3MB, contiguous, write-combined)
        fprintf(stderr, "[15] Allocating GPFIFO area...\n");
        uint64_t gpfifo_size = 0x300000;
        if (!alloc_gpu_mem(gpfifo_size, &h_gpfifo_mem, &gpfifo_gpu, &gpfifo_cpu,
                           true, NVOS32_ATTR_PHYSICALITY_CONTIGUOUS,
                           (NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED << 23))) {
            fprintf(stderr, "Failed to allocate GPFIFO memory\n");
            return false;
        }
        fprintf(stderr, "[15] GPFIFO area OK: gpu_va=0x%lx, cpu=%p\n",
                (unsigned long)gpfifo_gpu, gpfifo_cpu);
        memset(gpfifo_cpu, 0, gpfifo_size);

        // 16. Create context share
        fprintf(stderr, "[16] Creating context share (parent=0x%x, hVASpace=0x%x)...\n",
                h_channel_group, h_vaspace);
        NV_CTXSHARE_ALLOCATION_PARAMETERS ctx = {};
        ctx.hVASpace = h_vaspace;
        ctx.flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
        h_ctxshare = rm_alloc(h_channel_group, FERMI_CONTEXT_SHARE_A, &ctx, sizeof(ctx));
        if (!h_ctxshare) return false;

        // 17. Create compute GPFIFO channel
        uint32_t entries = 0x10000;
        h_compute_channel = create_gpfifo_channel(0, entries, &compute_token, true);
        if (!h_compute_channel) return false;

        // 18. Create copy (DMA) GPFIFO channel
        h_copy_channel = create_gpfifo_channel(0x100000, entries, &copy_token, false);
        if (!h_copy_channel) return false;

        // 19. Schedule channel group
        NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS sched = {};
        sched.bEnable = 1;
        rm_control(h_channel_group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &sched, sizeof(sched));

        // 20. Allocate command queue (2MB)
        NvHandle h_cmdq = 0;
        uint64_t cmdq_size = 0x200000;
        if (!alloc_gpu_mem(cmdq_size, &h_cmdq, &cmdq_gpu, &cmdq_cpu, true)) {
            fprintf(stderr, "Failed to allocate command queue\n");
            return false;
        }
        memset(cmdq_cpu, 0, cmdq_size);

        fprintf(stderr, "GPU init OK: compute_token=0x%x, copy_token=0x%x\n",
                compute_token, copy_token);
        return true;
    }

    // =========================================================================
    // GPU memory allocation
    // =========================================================================

    bool alloc_gpu_mem(uint64_t size, NvHandle* out_handle,
                       uint64_t* out_gpu_addr, void** out_cpu_ptr,
                       bool cpu_access,
                       NvU32 physicality = NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS,
                       NvU32 map_flags = 0) {
        uint64_t page_size = (size >= (8 << 20)) ? (2 << 20) : (4 << 10);
        size = (size + page_size - 1) & ~(page_size - 1);

        uint64_t va = alloc_va(size, page_size);

        bool huge = (page_size >= (2 << 20));

        NV_MEMORY_ALLOCATION_PARAMS mem = {};
        mem.owner = h_root;
        mem.type = NVOS32_TYPE_IMAGE;
        mem.flags = NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED
                  | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED
                  | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE
                  | NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT
                  | NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM;
        mem.format = 6;
        mem.size = size;
        mem.alignment = page_size;
        mem.offset = 0;
        mem.limit = size - 1;
        mem.attr = (physicality << 27)
                 | (huge ? (NVOS32_ATTR_PAGE_SIZE_HUGE << 23) : 0);
        mem.attr2 = (NVOS32_ATTR2_GPU_CACHEABLE_YES << 2)
                  | (huge ? (NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) : 0)
                  | NVOS32_ATTR2_ZBC_PREFER_NO_ZBC;

        NvHandle h_mem = rm_alloc(h_device, NV1_MEMORY_USER, &mem, sizeof(mem));
        if (!h_mem) return false;
        fprintf(stderr, "  alloc_gpu_mem: NV1_MEMORY_USER OK: h=0x%x\n", h_mem);

        void* cpu_ptr = nullptr;
        if (cpu_access) {
            cpu_ptr = map_to_cpu(h_device, h_mem, size, map_flags);
            if (!cpu_ptr) {
                fprintf(stderr, "CPU map failed for GPU mem\n");
                return false;
            }
            fprintf(stderr, "  alloc_gpu_mem: CPU map OK: %p\n", cpu_ptr);
        }

        // UVM external range
        UVM_CREATE_EXTERNAL_RANGE_PARAMS ext_range = {};
        ext_range.base = va;
        ext_range.length = size;
        int ext_ret = uvm_ioctl((UVM_CREATE_EXTERNAL_RANGE), &ext_range);
        fprintf(stderr, "  alloc_gpu_mem: UVM_CREATE_EXTERNAL_RANGE ret=%d\n", ext_ret);

        // DMA map (hDma = virtmem handle, per tinygrad)
        NVOS46_PARAMETERS dma = {};
        dma.hClient = h_root;
        dma.hDevice = h_device;
        dma.hDma = h_virtmem;
        dma.hMemory = h_mem;
        dma.length = size;
        dma.flags = (NVOS46_FLAGS_PAGE_SIZE_4KB << 8)
                  | (NVOS46_FLAGS_CACHE_SNOOP_ENABLE << 4)
                  | (NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE << 15);
        dma.dmaOffset = va;
        int dma_ret = nv_ioctl(fd_ctl, NV_ESC_RM_MAP_MEMORY_DMA, &dma, sizeof(dma));
        fprintf(stderr, "  alloc_gpu_mem: DMA map ret=%d status=0x%x\n", dma_ret, dma.status);

        // UVM map
        UVM_MAP_EXTERNAL_ALLOCATION_PARAMS uvm_map = {};
        uvm_map.base = va;
        uvm_map.length = size;
        uvm_map.offset = 0;
        uvm_map.gpuAttributesCount = 1;
        uvm_map.perGpuAttributes[0].gpuUuid = gpu_uuid;
        uvm_map.perGpuAttributes[0].gpuMappingType = 1; // DEFAULT
        uvm_map.rmCtrlFd = fd_ctl;
        uvm_map.hClient = h_root;
        uvm_map.hMemory = h_mem;
        int uvm_ret = uvm_ioctl((UVM_MAP_EXTERNAL_ALLOCATION), &uvm_map);
        fprintf(stderr, "  alloc_gpu_mem: UVM_MAP_EXTERNAL ret=%d\n", uvm_ret);

        *out_handle = h_mem;
        *out_gpu_addr = va;
        if (out_cpu_ptr) *out_cpu_ptr = cpu_ptr;
        return true;
    }

    // =========================================================================
    // High-level GPU memory API
    // =========================================================================

    struct GpuAlloc {
        uint64_t gpu_addr;
        void*    cpu_ptr;
        uint64_t size;
        NvHandle handle;
    };

    GpuAlloc gpu_malloc(uint64_t size) {
        GpuAlloc a = {};
        a.size = size;
        if (!alloc_gpu_mem(size, &a.handle, &a.gpu_addr, &a.cpu_ptr, true))
            fprintf(stderr, "gpu_malloc(%lu) failed\n", (unsigned long)size);
        return a;
    }

    // =========================================================================
    // GPFIFO channel creation
    // =========================================================================

    NvHandle create_gpfifo_channel(uint64_t offset, uint32_t entries,
                                    uint32_t* out_token, bool compute) {
        NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS params = {};
        params.gpFifoOffset = gpfifo_gpu + offset;
        params.gpFifoEntries = entries;
        params.hContextShare = h_ctxshare;
        params.hObjectBuffer = h_gpfifo_mem;
        params.hUserdMemory[0] = h_gpfifo_mem;
        params.userdOffset[0] = offset + entries * 8; // control area after ring

        NvHandle h_channel = rm_alloc(h_channel_group,
            BLACKWELL_CHANNEL_GPFIFO_A, &params, sizeof(params));
        if (!h_channel) return 0;

        if (compute) {
            h_compute_obj = rm_alloc(h_channel, BLACKWELL_COMPUTE_B);
            if (!h_compute_obj) return 0;
        } else {
            h_copy_obj = rm_alloc(h_channel, BLACKWELL_DMA_COPY_B);
            if (!h_copy_obj) return 0;
        }

        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS ws = {};
        ws.workSubmitToken = 0xFFFFFFFF;
        rm_control(h_channel, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                   &ws, sizeof(ws));
        *out_token = ws.workSubmitToken;

        // Register channel with UVM
        uint64_t uvm_base = alloc_va(0x4000000, 0x4000000);
        UVM_REGISTER_CHANNEL_PARAMS reg = {};
        reg.gpuUuid = gpu_uuid;
        reg.rmCtrlFd = fd_ctl;
        reg.hClient = h_root;
        reg.hChannel = h_channel;
        reg.base = uvm_base;
        reg.length = 0x4000000;
        uvm_ioctl((UVM_REGISTER_CHANNEL), &reg);

        return h_channel;
    }

    // =========================================================================
    // Command submission
    // =========================================================================

    uint32_t  cmdq_base = 0;      // base of current command block

    uint32_t* cmdq_ptr() {
        return (uint32_t*)((uint8_t*)cmdq_cpu + cmdq_offset);
    }

    // Start a new command block at a fresh cmdq offset.
    // GPU caches pushbuffer data by address — reusing the same offset
    // causes the GPU to re-execute stale commands from a prior dispatch.
    // We advance to a new 256-byte-aligned position each time.
    void cmd_begin() {
        // Align up to 256 bytes (GPU PB fetch granularity)
        cmdq_base = (cmdq_offset + 255) & ~255u;
        // Wrap around at 1.5MB to stay within 2MB cmdq, leaving room
        if (cmdq_base >= 0x180000) cmdq_base = 0;
        cmdq_offset = cmdq_base;
    }

    void cmd(uint32_t dw) {
        cmdq_ptr()[0] = dw;
        cmdq_offset += 4;
    }

    void nvm(int subchannel, uint32_t reg, uint32_t count) {
        cmd((1 << 29) | (count << 16) | (subchannel << 13) | (reg >> 2));
    }

    void nvm_data(uint32_t data) { cmd(data); }

    void submit_compute() {
        uint32_t n_dwords = (cmdq_offset - cmdq_base) / 4;
        submit_gpfifo(cmdq_gpu + cmdq_base, n_dwords, 0, compute_token, compute_put);
        compute_put++;
    }

    void submit_gpfifo(uint64_t addr, uint32_t n_dwords,
                       uint64_t ring_offset, uint32_t token, uint32_t& put) {
        uint32_t entries = 0x10000;
        volatile uint64_t* ring = (volatile uint64_t*)((uint8_t*)gpfifo_cpu + ring_offset);

        // GPFIFO entry: GP_ENTRY0 bits 31:2 = addr, GP_ENTRY1 bits 7:0 = addr_hi,
        // GP_ENTRY1 bits 30:10 = length_dw, GP_ENTRY1 bit 9 = LEVEL (0=MAIN)
        uint64_t entry = ((addr >> 2) << 2) | ((uint64_t)n_dwords << 42);
        ring[put % entries] = entry;

        volatile uint32_t* gp_put = (volatile uint32_t*)((uint8_t*)gpfifo_cpu +
            ring_offset + entries * 8 + 0x8C);
        *gp_put = (put + 1) % entries;

        __sync_synchronize();
        gpu_mmio[0x90 / 4] = token;
    }

    // =========================================================================
    // Channel one-time setup (SET_OBJECT, memory windows)
    // =========================================================================

    bool channel_setup_done = false;

    void setup_channels() {
        if (channel_setup_done) return;

        cmd_begin();

        // SET_OBJECT on subchannel 1 = BLACKWELL_COMPUTE_B (0xCEC0)
        nvm(1, 0x0000, 1); nvm_data(0xCEC0);

        // SET_SHADER_LOCAL_MEMORY_WINDOW (subchannel 1, method 0x07B0)
        uint64_t local_window = 0x729300000000ULL;
        nvm(1, 0x07B0, 2);
        nvm_data((uint32_t)(local_window >> 32));
        nvm_data((uint32_t)(local_window & 0xFFFFFFFF));

        // SET_SHADER_SHARED_MEMORY_WINDOW (subchannel 1, method 0x02A0)
        uint64_t shared_window = 0x729400000000ULL;
        nvm(1, 0x02A0, 2);
        nvm_data((uint32_t)(shared_window >> 32));
        nvm_data((uint32_t)(shared_window & 0xFFFFFFFF));

        // SET_OBJECT on subchannel 4 = BLACKWELL_DMA_COPY_B (0xCAB5)
        nvm(4, 0x0000, 1); nvm_data(0xCAB5);

        submit_compute();

        channel_setup_done = true;
    }

    // =========================================================================
    // Semaphore for GPU synchronization
    // =========================================================================

    GpuAlloc sem_alloc = {};
    uint64_t sem_counter = 0;

    void init_semaphore() {
        if (sem_alloc.cpu_ptr) return;
        sem_alloc = gpu_malloc(4096);
        memset(sem_alloc.cpu_ptr, 0, 4096);
    }

    // =========================================================================
    // CUBIN upload — upload entire ELF image to GPU memory
    // =========================================================================

    GpuAlloc cubin_gpu = {};

    uint64_t upload_cubin(const uint8_t* image, size_t size) {
        if (cubin_gpu.cpu_ptr) return cubin_gpu.gpu_addr;
        cubin_gpu = gpu_malloc(size);
        memcpy(cubin_gpu.cpu_ptr, image, size);
        __sync_synchronize();
        return cubin_gpu.gpu_addr;
    }

    // =========================================================================
    // Kernel launch
    // =========================================================================

    // -------------------------------------------------------------------------
    // Single-pushbuffer dispatch model
    //
    // Blackwell's GPFIFO controller stalls permanently after processing a
    // pushbuffer entry that contains SEND_PCAS.  No subsequent GPFIFO entries
    // are processed (kernel dispatches are silently dropped).
    //
    // The workaround: put ALL dispatches AND the HOST semaphore release into
    // a SINGLE pushbuffer, submitted as ONE GPFIFO entry.  Within a single
    // pushbuffer, multiple SEND_PCAS dispatches + HOST semaphore all work.
    //
    // API:
    //   begin_commands()              — start a new pushbuffer
    //   launch_kernel(...)            — append SEND_PCAS to the pushbuffer
    //   wait_kernel()                 — append SEM_RELEASE, submit, poll, sleep
    // -------------------------------------------------------------------------

    bool pb_started = false;

    void begin_commands() {
        init_semaphore();
        cmd_begin();
        pb_started = true;

        if (!channel_setup_done) {
            nvm(1, 0x0000, 1); nvm_data(0xCEC0);  // SET_OBJECT subchannel 1 = COMPUTE
            nvm(4, 0x0000, 1); nvm_data(0xCAB5);  // SET_OBJECT subchannel 4 = DMA_COPY
            channel_setup_done = true;
        }

        // Memory windows
        uint64_t local_window = 0x729300000000ULL;
        nvm(1, 0x07B0, 2);
        nvm_data((uint32_t)(local_window >> 32));
        nvm_data((uint32_t)(local_window & 0xFFFFFFFF));

        uint64_t shared_window = 0x729400000000ULL;
        nvm(1, 0x02A0, 2);
        nvm_data((uint32_t)(shared_window >> 32));
        nvm_data((uint32_t)(shared_window & 0xFFFFFFFF));
    }

    void launch_kernel(uint64_t code_gpu_addr, uint32_t code_size,
                       uint32_t reg_count, uint32_t shared_mem,
                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                       uint64_t cbuf0_gpu_addr, uint32_t cbuf0_size) {
        // Auto-start pushbuffer if needed
        if (!pb_started) begin_commands();

        // Build QMD (CUDA-captured template for SM120)
        QMD qmd = {};
        memset(&qmd, 0, sizeof(qmd));

        qmd.dw[4] = 0x013f0000;   // qmd_type=GRID_CTA, group_id=0x3F
        qmd.dw[9] = 0x00000000;   // QMD release disabled
        qmd.dw[10] = 0x00190000;
        qmd.dw[12] = 0x02070100;
        qmd.dw[14] = 0x2f5003a4;  // sass_version=0xa4, sampler_index=1, major=5
        qmd.dw[19] = 0x80610000;  // cwd_membar_type=1
        qmd.dw[20] = 0x00000008;
        qmd.dw[22] = 0x04000000;
        qmd.dw[58] = 0x00000011;  // cbuf0+cbuf1 valid

        // Program address
        uint64_t prog_s4 = code_gpu_addr >> 4;
        qmd.dw[32] = (uint32_t)(prog_s4 & 0xFFFFFFFF);
        qmd.dw[33] = (uint32_t)(prog_s4 >> 32) & 0x1FFFFF;

        // Block dimensions + register count + barrier
        qmd.dw[34] = (block_y << 16) | block_x;
        uint32_t regs = (reg_count + 1) & ~1u;
        if (regs < 16) regs = 16;
        qmd.dw[35] = block_z | (regs << 8) | (1 << 17);

        // Shared memory (round UP to 128-byte units)
        uint32_t smem_s7 = (shared_mem + 127) >> 7;
        if (smem_s7 < 8) smem_s7 = 8;
        // SMEM_CONFIG: min(tier_kb*1024 for tiers in [32,64,100] if >= bytes) / 4096 + 1
        uint32_t smem_cfg = 9;   // 32KB tier
        if (shared_mem > 32768) smem_cfg = 17;   // 64KB tier
        if (shared_mem > 65536) smem_cfg = 26;   // 100KB tier
        qmd.dw[36] = smem_s7 | (smem_cfg << 11) | (0x1A << 17) | (smem_cfg << 23);

        // Grid dimensions
        qmd.dw[39] = grid_x;
        qmd.dw[40] = grid_y;
        qmd.dw[41] = grid_z;

        // Constant buffer 0
        uint64_t cb_addr_s6 = cbuf0_gpu_addr >> 6;
        uint32_t cb_size_s4 = (cbuf0_size + 15) / 16;
        qmd.dw[42] = (uint32_t)(cb_addr_s6 & 0xFFFFFFFF);
        qmd.dw[43] = ((uint32_t)(cb_addr_s6 >> 32) & 0x7FFFF) | (cb_size_s4 << 19);

        // Program prefetch
        qmd.dw[59] = (uint32_t)(code_gpu_addr >> 8);
        qmd.dw[60] = (uint32_t)(code_gpu_addr >> 40);

        // Upload QMD to GPU
        GpuAlloc qmd_mem = gpu_malloc(4096);
        memcpy(qmd_mem.cpu_ptr, &qmd, sizeof(qmd));
        __sync_synchronize();

        // Append SEND_PCAS to current pushbuffer (no submit yet)
        nvm(1, 0x02B4, 1); nvm_data((uint32_t)(qmd_mem.gpu_addr >> 8));
        nvm(1, 0x02C0, 1); nvm_data(9);
    }

    void wait_kernel() {
        if (!pb_started) return;

        // Append HOST semaphore release to the SAME pushbuffer
        sem_counter++;
        nvm(0, 0x005C, 5);
        nvm_data((uint32_t)(sem_alloc.gpu_addr & 0xFFFFFFFF));
        nvm_data((uint32_t)(sem_alloc.gpu_addr >> 32));
        nvm_data((uint32_t)(sem_counter & 0xFFFFFFFF));
        nvm_data((uint32_t)(sem_counter >> 32));
        nvm_data(0x01000001);  // SEM_EXECUTE: RELEASE + 64BIT

        // Submit entire pushbuffer as ONE GPFIFO entry
        submit_compute();
        pb_started = false;

        // Poll semaphore
        volatile uint64_t* sem64 = (volatile uint64_t*)sem_alloc.cpu_ptr;
        int spins = 0;
        while (*sem64 < sem_counter) {
            __sync_synchronize();
            if (++spins > 100000000) {
                fprintf(stderr, "wait_kernel: timeout! sem64=0x%lx expected>=0x%lx\n",
                        (unsigned long)*sem64, (unsigned long)sem_counter);
                break;
            }
        }

        // HOST semaphore fires before kernel completion (no WFI available).
        // Small sleep ensures all dispatched kernels finish writing output.
        usleep(100);
        __sync_synchronize();
    }

    // =========================================================================
    // High-level kernel launch with cbuf0 preparation
    // =========================================================================

    // Cbuf0 layout for Blackwell:
    //   [0, param_base)         = driver template (incl. grid dims, windows, etc.)
    //   [param_base, cbuf0_size) = kernel arguments
    // Driver params at fixed DWORD offsets within template:
    //   DWORD[188-189] = shared_mem_window (uint64 LE)
    //   DWORD[190-191] = local_mem_window (uint64 LE)
    //   DWORD[223]     = 0xfffdc0

    GpuAlloc prepare_cbuf0(const void* args, uint32_t args_size,
                           uint32_t cbuf0_size, uint32_t param_base,
                           uint32_t block_x = 1, uint32_t block_y = 1,
                           uint32_t block_z = 1) {
        if (cbuf0_size < 896) cbuf0_size = 896;
        // Ensure cbuf0 is large enough for param_base + args
        uint32_t needed = param_base + args_size;
        if (cbuf0_size < needed) cbuf0_size = (needed + 15) & ~15u;

        GpuAlloc cbuf = gpu_malloc(cbuf0_size);
        memset(cbuf.cpu_ptr, 0, cbuf0_size);

        // Copy kernel arguments at param_base offset
        memcpy((uint8_t*)cbuf.cpu_ptr + param_base, args, args_size);

        // Set driver params in template area
        uint32_t* dw = (uint32_t*)cbuf.cpu_ptr;
        uint64_t shared_window = 0x729400000000ULL;
        uint64_t local_window  = 0x729300000000ULL;
        dw[188] = (uint32_t)(shared_window & 0xFFFFFFFF);
        dw[189] = (uint32_t)(shared_window >> 32);
        dw[190] = (uint32_t)(local_window & 0xFFFFFFFF);
        dw[191] = (uint32_t)(local_window >> 32);

        // Block dimensions at 0x360-0x368 (kernel reads via LDC c[0x0][0x360])
        dw[216] = block_x;   // offset 0x360
        dw[217] = block_y;   // offset 0x364
        dw[218] = block_z;   // offset 0x368

        dw[223] = 0xfffdc0;  // offset 0x37c (stack/misc)

        __sync_synchronize();
        return cbuf;
    }
};
