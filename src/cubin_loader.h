// cubin_loader.h — CUBIN ELF parser for cudaless kernel loading
//
// Parses a CUBIN file (an ELF with CUDA-specific sections) to extract
// per-kernel metadata: code offset/size, register count, shared memory,
// constant buffer 0 size. Uses only <elf.h> (Linux system header).
//
// Reference: tinygrad ops_nv.py _parse_elf_info()

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <string>
#include <unordered_map>
#include <vector>

struct CubinKernel {
    std::string name;           // demangled-ish kernel name
    std::string mangled_name;   // full mangled symbol name

    // Code section
    uint64_t code_offset;       // offset within ELF image
    uint64_t code_size;

    // From .nv.info.<name>
    uint32_t reg_count;
    uint32_t cbuf0_size;        // total cbuf0 size (from .nv.constant0 section)
    uint32_t param_base;        // offset where kernel args start in cbuf0 (from EIATTR_PARAM_CBANK)
    uint32_t local_mem_low;
    uint32_t local_mem_high;

    // From .nv.shared.<name>
    uint32_t shared_mem_size;

    // Section index in ELF (for relocation mapping)
    uint32_t text_shndx;
};

struct CubinReloc {
    uint32_t target_shndx;      // section to apply reloc in
    uint64_t offset;            // offset within target section
    uint32_t type;              // relocation type
    uint32_t sym_idx;           // symbol index
    int64_t  addend;            // addend
};

class CubinLoader {
public:
    std::vector<uint8_t> image;
    std::vector<CubinKernel> kernels;
    std::vector<CubinReloc> relocs;

    // Symbol table for relocations
    struct SymInfo {
        std::string name;
        uint64_t value;
        uint32_t shndx;
    };
    std::vector<SymInfo> symbols;

    // Global .nv.constant3 — SM120 uses cbuf3 for __constant__ data
    uint64_t cbuf3_offset = 0;  // offset within ELF image
    uint32_t cbuf3_size = 0;    // size in bytes (0 if none)

    // Section offsets (ELF file offsets, for computing GPU addresses after upload)
    std::unordered_map<uint32_t, uint64_t> section_file_offsets;
    std::unordered_map<uint32_t, uint64_t> section_sizes;

    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { perror(path); return false; }
        fseek(f, 0, SEEK_END);
        size_t sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        image.resize(sz);
        if (fread(image.data(), 1, sz, f) != sz) { fclose(f); return false; }
        fclose(f);
        return parse();
    }

    CubinKernel* find_kernel(const char* name) {
        // First try exact match on demangled name
        for (auto& k : kernels)
            if (k.name == name)
                return &k;
        // Fallback to substring match on mangled name
        for (auto& k : kernels)
            if (k.mangled_name.find(name) != std::string::npos)
                return &k;
        return nullptr;
    }

private:
    const Elf64_Ehdr* ehdr() const { return (const Elf64_Ehdr*)image.data(); }

    const Elf64_Shdr* shdr(int i) const {
        return (const Elf64_Shdr*)(image.data() + ehdr()->e_shoff + i * ehdr()->e_shentsize);
    }

    const char* shstrtab() const {
        return (const char*)(image.data() + shdr(ehdr()->e_shstrndx)->sh_offset);
    }

    const char* section_name(int i) const {
        return shstrtab() + shdr(i)->sh_name;
    }

    bool parse() {
        auto* e = ehdr();
        if (memcmp(e->e_ident, ELFMAG, SELFMAG) != 0) {
            fprintf(stderr, "cubin_loader: not an ELF file\n");
            return false;
        }

        // Pass 1: build section name -> index map, record offsets
        std::unordered_map<std::string, int> sec_by_name;
        for (int i = 0; i < e->e_shnum; i++) {
            sec_by_name[section_name(i)] = i;
            section_file_offsets[i] = shdr(i)->sh_offset;
            section_sizes[i] = shdr(i)->sh_size;
        }

        // Pass 2: parse symbol table
        for (int i = 0; i < e->e_shnum; i++) {
            auto* sh = shdr(i);
            if (sh->sh_type != SHT_SYMTAB) continue;
            const char* strtab = (const char*)(image.data() + shdr(sh->sh_link)->sh_offset);
            int nsym = sh->sh_size / sh->sh_entsize;
            symbols.resize(nsym);
            for (int j = 0; j < nsym; j++) {
                auto* sym = (const Elf64_Sym*)(image.data() + sh->sh_offset + j * sh->sh_entsize);
                symbols[j].name = strtab + sym->st_name;
                symbols[j].value = sym->st_value;
                symbols[j].shndx = sym->st_shndx;
            }
        }

        // Pass 3: find kernel .text sections and build kernel list
        std::unordered_map<std::string, int> kernel_by_mangled;
        for (int i = 0; i < e->e_shnum; i++) {
            std::string sname = section_name(i);
            if (sname.substr(0, 6) != ".text.") continue;
            auto* sh = shdr(i);
            if (sh->sh_size == 0) continue;

            CubinKernel k = {};
            k.mangled_name = sname.substr(6);
            k.name = demangle_simple(k.mangled_name);
            k.code_offset = sh->sh_offset;
            k.code_size = sh->sh_size;
            k.text_shndx = i;

            kernel_by_mangled[k.mangled_name] = kernels.size();
            kernels.push_back(std::move(k));
        }

        // Pass 4: parse .nv.info.<name> sections for register count, cbuf0 size
        for (int i = 0; i < e->e_shnum; i++) {
            std::string sname = section_name(i);
            if (sname.substr(0, 9) != ".nv.info.") continue;
            std::string kname = sname.substr(9);
            auto it = kernel_by_mangled.find(kname);
            if (it == kernel_by_mangled.end()) continue;
            auto& k = kernels[it->second];

            const uint8_t* data = image.data() + shdr(i)->sh_offset;
            uint64_t size = shdr(i)->sh_size;
            parse_nv_info(data, size, k);
        }

        // Pass 4b: parse global .nv.info for register counts (Blackwell stores them here)
        for (int i = 0; i < e->e_shnum; i++) {
            if (std::string(section_name(i)) != ".nv.info") continue;
            const uint8_t* data = image.data() + shdr(i)->sh_offset;
            uint64_t size = shdr(i)->sh_size;
            parse_global_nv_info(data, size, kernel_by_mangled);
        }

        // Pass 5: parse .nv.shared.<name> for shared memory size
        for (int i = 0; i < e->e_shnum; i++) {
            std::string sname = section_name(i);
            if (sname.substr(0, 11) != ".nv.shared.") continue;
            std::string kname = sname.substr(11);
            auto it = kernel_by_mangled.find(kname);
            if (it == kernel_by_mangled.end()) continue;
            kernels[it->second].shared_mem_size = shdr(i)->sh_size;
        }

        // Pass 6: parse .nv.constant0.<name> for total cbuf0 size
        for (int i = 0; i < e->e_shnum; i++) {
            std::string sname = section_name(i);
            if (sname.substr(0, 14) != ".nv.constant0.") continue;
            std::string kname = sname.substr(14);
            auto it = kernel_by_mangled.find(kname);
            if (it == kernel_by_mangled.end()) continue;
            auto& k = kernels[it->second];
            // .nv.constant0 section size = total cbuf0 (template + params)
            k.cbuf0_size = shdr(i)->sh_size;
        }

        // Pass 7: detect global .nv.constant3 (SM120 __constant__ data)
        for (int i = 0; i < e->e_shnum; i++) {
            if (section_name(i) == ".nv.constant3") {
                cbuf3_offset = shdr(i)->sh_offset;
                cbuf3_size = shdr(i)->sh_size;
                break;
            }
        }

        // Pass 8: collect relocations
        for (int i = 0; i < e->e_shnum; i++) {
            auto* sh = shdr(i);
            if (sh->sh_type != SHT_RELA) continue;

            uint32_t target = sh->sh_info;
            int nrel = sh->sh_size / sh->sh_entsize;
            for (int j = 0; j < nrel; j++) {
                auto* rel = (const Elf64_Rela*)(image.data() + sh->sh_offset + j * sh->sh_entsize);
                CubinReloc r = {};
                r.target_shndx = target;
                r.offset = rel->r_offset;
                r.type = ELF64_R_TYPE(rel->r_info);
                r.sym_idx = ELF64_R_SYM(rel->r_info);
                r.addend = rel->r_addend;
                relocs.push_back(r);
            }
        }

        return true;
    }

    void parse_global_nv_info(const uint8_t* data, uint64_t size,
                              const std::unordered_map<std::string, int>& kernel_by_mangled) {
        // Global .nv.info has fmt=0x04 entries with 8-byte payload:
        //   [sym_index(4B), value(4B)]
        uint64_t off = 0;
        while (off + 4 <= size) {
            uint8_t fmt = data[off];
            uint8_t param = data[off + 1];
            uint16_t val16 = *(const uint16_t*)(data + off + 2);

            if (fmt == 0x04) {
                if (val16 >= 8 && off + 4 + val16 <= size) {
                    uint32_t sym_idx = *(const uint32_t*)(data + off + 4);
                    uint32_t value   = *(const uint32_t*)(data + off + 8);

                    if (sym_idx < symbols.size()) {
                        auto it = kernel_by_mangled.find(symbols[sym_idx].name);
                        if (it != kernel_by_mangled.end()) {
                            auto& k = kernels[it->second];
                            if (param == 0x2F) k.reg_count = value;
                        }
                    }
                }
                off += 4 + val16;
            } else {
                off += 4;
            }
        }
    }

    void parse_nv_info(const uint8_t* data, uint64_t size, CubinKernel& k) {
        // Format from tinygrad _parse_elf_info:
        //   type  = byte at offset+0
        //   param = byte at offset+1  (high byte) combined with uint16 at offset+2
        //   But actually: format = u8 type, u8 param_encoding, u16 value_or_size
        //   if type == 0x04: data follows (size bytes), offset += 4 + size
        //   else: the u16 at offset+2 IS the data, offset += 4

        uint64_t off = 0;
        while (off + 4 <= size) {
            uint8_t format = data[off];
            uint8_t param = data[off + 1];
            uint16_t val16 = *(const uint16_t*)(data + off + 2);

            if (format == 0x04) {
                // Variable-length data follows
                const uint8_t* payload = data + off + 4;
                uint32_t payload_size = val16;

                if (param == 0x0A && payload_size >= 6) {
                    // EIATTR_PARAM_CBANK: u16 sym_idx, u16 ordinal, u16 param_base
                    // param_base = offset within cbuf0 where kernel args start
                    uint16_t param_base = *(const uint16_t*)(payload + 4);
                    k.param_base = param_base;
                }
                if (param == 0x12 && payload_size >= 8) {
                    // EIATTR_MIN_STACK_SIZE
                    uint32_t local_low = *(const uint32_t*)(payload);
                    uint32_t local_high = *(const uint32_t*)(payload + 4);
                    k.local_mem_low = local_low + 0x240;
                    k.local_mem_high = local_high;
                }

                off += 4 + payload_size;
            } else {
                // Fixed 4-byte entry, val16 is the data
                if (param == 0x2F) {
                    // EIATTR_REGCOUNT
                    k.reg_count = val16;
                }

                off += 4;
            }
        }
    }

    static std::string demangle_simple(const std::string& mangled) {
        // Extract the kernel name from C++ mangled symbol
        // _Z<len><name>... -> name
        if (mangled.size() < 3 || mangled[0] != '_' || mangled[1] != 'Z')
            return mangled;
        size_t pos = 2;
        int len = 0;
        while (pos < mangled.size() && mangled[pos] >= '0' && mangled[pos] <= '9') {
            len = len * 10 + (mangled[pos] - '0');
            pos++;
        }
        if (len > 0 && pos + len <= mangled.size())
            return mangled.substr(pos, len);
        return mangled;
    }
};
