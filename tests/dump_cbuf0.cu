// dump_cbuf0.cu — Dump the cbuf0 template from the cubin's .nv.constant0 section
// and compare against what the cudaless loader constructs.
// This tells us exactly what bytes differ.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// Parse cubin ELF to extract .nv.constant0 section data and param_base for a kernel
struct CubinInfo {
    std::vector<uint8_t> constant0_data;
    uint32_t param_base;
    uint32_t cbuf0_size;
    std::string kernel_name;
};

static CubinInfo parse_cubin_constant0(const char* path, const char* match) {
    CubinInfo info = {};
    FILE* f = fopen(path, "rb");
    if (!f) return info;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(sz);
    fread(data.data(), 1, sz, f);
    fclose(f);

    // Minimal ELF parsing
    auto r16 = [&](long off) -> uint16_t { uint16_t v; memcpy(&v, data.data()+off, 2); return v; };
    auto r32 = [&](long off) -> uint32_t { uint32_t v; memcpy(&v, data.data()+off, 4); return v; };
    auto r64 = [&](long off) -> uint64_t { uint64_t v; memcpy(&v, data.data()+off, 8); return v; };

    uint64_t e_shoff = r64(40);
    uint16_t e_shentsize = r16(58);
    uint16_t e_shnum = r16(60);
    uint16_t e_shstrndx = r16(62);

    uint64_t shstr_off = r64(e_shoff + e_shstrndx * e_shentsize + 24);

    auto sec_name = [&](int i) -> std::string {
        uint32_t name_off = r32(e_shoff + i * e_shentsize);
        const char* s = (const char*)data.data() + shstr_off + name_off;
        return std::string(s);
    };
    auto sec_offset = [&](int i) -> uint64_t { return r64(e_shoff + i * e_shentsize + 24); };
    auto sec_size = [&](int i) -> uint64_t { return r64(e_shoff + i * e_shentsize + 32); };
    auto sec_type = [&](int i) -> uint32_t { return r32(e_shoff + i * e_shentsize + 4); };
    auto sec_link = [&](int i) -> uint32_t { return r32(e_shoff + i * e_shentsize + 40); };
    auto sec_entsize = [&](int i) -> uint64_t { return r64(e_shoff + i * e_shentsize + 56); };

    // Find symtab
    uint64_t symtab_off = 0, symtab_entsize = 0;
    uint64_t strtab_off = 0;
    for (int i = 0; i < e_shnum; i++) {
        if (sec_type(i) == 2) { // SHT_SYMTAB
            symtab_off = sec_offset(i);
            symtab_entsize = sec_entsize(i);
            strtab_off = sec_offset(sec_link(i));
        }
    }

    // Find matching kernel's .nv.constant0 and .nv.info
    std::string target_mangled;
    for (int i = 0; i < e_shnum; i++) {
        std::string name = sec_name(i);
        if (name.substr(0, 6) != ".text.") continue;
        std::string mangled = name.substr(6);
        if (mangled.find(match) == std::string::npos) continue;
        if (mangled.find("GemmBatched") != std::string::npos) continue;
        if (mangled.find("Congruous") != std::string::npos) continue;
        target_mangled = mangled;
        info.kernel_name = mangled;
        break;
    }
    if (target_mangled.empty()) return info;

    // Get .nv.constant0.<kernel>
    std::string c0_name = ".nv.constant0." + target_mangled;
    for (int i = 0; i < e_shnum; i++) {
        if (sec_name(i) == c0_name) {
            uint64_t off = sec_offset(i);
            uint64_t size = sec_size(i);
            info.cbuf0_size = size;
            info.constant0_data.assign(data.data() + off, data.data() + off + size);
            break;
        }
    }

    // Get param_base from .nv.info.<kernel>
    std::string info_name = ".nv.info." + target_mangled;
    for (int i = 0; i < e_shnum; i++) {
        if (sec_name(i) == info_name) {
            const uint8_t* p = data.data() + sec_offset(i);
            uint64_t sz2 = sec_size(i);
            uint64_t off = 0;
            while (off + 4 <= sz2) {
                uint8_t fmt = p[off];
                uint8_t param = p[off+1];
                uint16_t val16; memcpy(&val16, p+off+2, 2);
                if (fmt == 0x04) {
                    if (param == 0x0A && val16 >= 6) {
                        uint16_t pb; memcpy(&pb, p+off+4+4, 2);
                        info.param_base = pb;
                    }
                    off += 4 + val16;
                } else {
                    off += 4;
                }
            }
            break;
        }
    }

    return info;
}

int main() {
    // Parse the CUTLASS cubin
    auto info = parse_cubin_constant0("cutlass_gemm.cubin", "64ELi64ELi64E");
    if (info.constant0_data.empty()) {
        printf("Failed to find kernel constant0 data\n");
        return 1;
    }

    printf("Kernel: %.80s...\n", info.kernel_name.c_str());
    printf("cbuf0_size = %u (0x%x)\n", info.cbuf0_size, info.cbuf0_size);
    printf("param_base = %u (0x%x)\n", info.param_base, info.param_base);
    printf("\n");

    // Build what the cudaless loader would construct
    std::vector<uint8_t> cudaless_cbuf0(info.cbuf0_size, 0);
    uint32_t* dw = (uint32_t*)cudaless_cbuf0.data();
    uint64_t shared_window = 0x729400000000ULL;
    uint64_t local_window  = 0x729300000000ULL;
    dw[188] = (uint32_t)(shared_window & 0xFFFFFFFF);
    dw[189] = (uint32_t)(shared_window >> 32);
    dw[190] = (uint32_t)(local_window & 0xFFFFFFFF);
    dw[191] = (uint32_t)(local_window >> 32);
    dw[216] = 128;  // example block_x
    dw[217] = 1;    // block_y
    dw[218] = 1;    // block_z
    dw[223] = 0xfffdc0;

    // Compare template region (0 to param_base) between cubin's .nv.constant0 and cudaless
    printf("=== Template region diffs (0 to 0x%x) ===\n", info.param_base);
    printf("Cubin .nv.constant0 vs cudaless memset(0)+template:\n\n");

    const uint8_t* cubin_data = info.constant0_data.data();
    int diff_count = 0;
    for (uint32_t off = 0; off < info.param_base && off < info.constant0_data.size(); off += 4) {
        uint32_t cubin_val, cudaless_val;
        memcpy(&cubin_val, cubin_data + off, 4);
        memcpy(&cudaless_val, cudaless_cbuf0.data() + off, 4);
        if (cubin_val != 0) {  // Show all non-zero values in cubin's template
            printf("  [0x%03x] dw[%3d]  cubin=0x%08x  cudaless=0x%08x  %s\n",
                   off, off/4, cubin_val, cudaless_val,
                   cubin_val == cudaless_val ? "OK" : "DIFF!");
            if (cubin_val != cudaless_val) diff_count++;
        }
    }
    printf("\nTotal template diffs: %d\n", diff_count);

    // Also show non-zero values in params region
    printf("\n=== Params region (0x%x to 0x%x): first 20 non-zero DWORDs ===\n",
           info.param_base, info.cbuf0_size);
    int shown = 0;
    for (uint32_t off = info.param_base; off < info.constant0_data.size() && shown < 20; off += 4) {
        uint32_t v;
        memcpy(&v, cubin_data + off, 4);
        if (v != 0) {
            printf("  [0x%03x] dw[%3d] = 0x%08x\n", off, off/4, v);
            shown++;
        }
    }

    // Also parse and compare the simple kernels cubin
    printf("\n\n=== SIMPLE KERNEL (for comparison) ===\n");
    auto info2 = parse_cubin_constant0("kernels.cubin", "mel_normalize");
    if (!info2.constant0_data.empty()) {
        printf("Kernel: %.80s...\n", info2.kernel_name.c_str());
        printf("cbuf0_size = %u (0x%x)\n", info2.cbuf0_size, info2.cbuf0_size);
        printf("param_base = %u (0x%x)\n", info2.param_base, info2.param_base);
        printf("\nTemplate non-zero values:\n");
        const uint8_t* d2 = info2.constant0_data.data();
        for (uint32_t off = 0; off < info2.param_base && off < info2.constant0_data.size(); off += 4) {
            uint32_t v; memcpy(&v, d2 + off, 4);
            if (v != 0) printf("  [0x%03x] dw[%3d] = 0x%08x\n", off, off/4, v);
        }
    }

    return 0;
}
