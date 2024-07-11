package sppark

// #cgo linux LDFLAGS: -ldl -Wl,-rpath,"$ORIGIN"
//
// #ifndef GO_CGO_EXPORT_PROLOGUE_H
// #ifdef _WIN32
// # include <windows.h>
// # include <stdio.h>
// #else
// # include <dlfcn.h>
// # include <errno.h>
// #endif
// #include <string.h>
// #include <stdlib.h>
// #ifdef __SPPARK_CGO_DEBUG__
// # include <stdio.h>
// #endif
//
// #include "cgo_sppark.h"
//
// void toGoString(_GoString_ *, char *);
//
// void toGoError(GoError *go_err, Error c_err)
// {
//     go_err->code = c_err.code;
//     if (c_err.message != NULL) {
//         toGoString(&go_err->message, c_err.message);
//         free(c_err.message);
//         c_err.message = NULL;
//     }
// }
//
// typedef struct {
//     void *ptr;
// } gpu_ptr_t;
//
// #define WRAP(ret_t, func, ...) __attribute__((section("_sppark"), used)) \
//	static struct { ret_t (*call)(__VA_ARGS__); const char *name; } \
//	func = { NULL, #func }; \
//	static ret_t go_##func(__VA_ARGS__)
//
// WRAP(gpu_ptr_t, clone_gpu_ptr_t, gpu_ptr_t *ptr)
// {   return (*clone_gpu_ptr_t.call)(ptr);   }
//
// WRAP(void, drop_gpu_ptr_t, gpu_ptr_t *ptr)
// {   (*drop_gpu_ptr_t.call)(ptr);   }
//
// WRAP(_Bool, cuda_available, void)
// {   return (*cuda_available.call)();   }
//
// typedef struct {
//     void *value;
//     const char *name;
// } dlsym_t;
//
// static _Bool go_load(_GoString_ *err, _GoString_ so_name)
// {
//     static void *hmod = NULL;
//     void *h;
//
//     if ((h = hmod) == NULL) {
//         size_t len = _GoStringLen(so_name);
//         char fname[len + 1];
//
//         memcpy(fname, _GoStringPtr(so_name), len);
//         fname[len] = '\0';
// #ifdef _WIN32
//         h = LoadLibraryA(fname);
// #else
//         h = dlopen(fname, RTLD_NOW|RTLD_GLOBAL);
// #endif
//         if ((hmod = h) != NULL) {
//             extern dlsym_t __start__sppark, __stop__sppark;
//             dlsym_t *sym;
//
//             for (sym = &__start__sppark; sym < &__stop__sppark; sym++) {
// #ifdef _WIN32
//                 sym->value = GetProcAddress(h, sym->name);
// #else
//                 sym->value = dlsym(h, sym->name);
// #endif
//                 if (sym->value == NULL) {
//                     h = NULL;
//                     break;
//                 }
// #ifdef __SPPARK_CGO_DEBUG__
//                 printf("%p %s\n", sym->value, sym->name);
// #endif
//             }
//         }
//     }
//
//     if (h == NULL) {
// #ifdef _WIN32
//         static char buf[24];
//         snprintf(buf, sizeof(buf), "WIN32 Error #0x%x", GetLastError());
//         toGoString(err, buf);
//         if (hmod) FreeLibrary(hmod);
// #else
//         toGoString(err, dlerror());
//         if (hmod) dlclose(hmod);
// #endif
//         hmod = h;
//     }
//
//     return h != NULL;
// }
// #endif
import "C"

import (
    blst "github.com/supranational/blst/build"
    "io"
    "log"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "strings"
)

//export toGoString
func toGoString(go_str *string, c_str *C.char) {
    *go_str = C.GoString(c_str)
}

var SrcRoot string

func init() {
    if _, self, _, ok := runtime.Caller(0); ok {
        SrcRoot = filepath.Dir(filepath.Dir(self))
    }
}

func Load(baseName string, options ...string) {
    baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))

    var dst, src string

    if exe, err := os.Executable(); err == nil {
        dst = filepath.Join(filepath.Dir(exe), filepath.Base(baseName))
        if runtime.GOOS == "windows" {
            dst += ".dll"
        } else {
            dst += ".so"
        }
    } else {
        log.Panic(err)
    }

    if _, caller, _, ok := runtime.Caller(1); ok {
        src = filepath.Join(filepath.Dir(caller), baseName + ".cu")
    } else {
        log.Panic("passed the event horizon")
    }

    // To facilitate the edit-compile-run turnaround check if the source
    // .cu file is writable and see if it's newer than the destination
    // shared object...
    rebuild := false
    if fd, err := os.OpenFile(src, os.O_RDWR, 0); err == nil {
        src_stat, _ := fd.Stat()
        fd.Close()
        dst_stat, err := os.Stat(dst)
        rebuild = err != nil || src_stat.ModTime().After(dst_stat.ModTime())
    }

    var go_err string

    if rebuild || !bool(C.go_load(&go_err, dst)) {
        if !build(dst, src, options...) {
            log.Panic("failed to build the shared object")
        }
        go_err = ""
        if !C.go_load(&go_err, dst) {
           log.Panic(go_err)
        }
    }
}

func build(dst string, src string, custom_args ...string) bool {
    var args []string

    args = append(args, "-shared", "-o", dst, src)
    args = append(args, "-I" + SrcRoot)
    args = append(args, filepath.Join(SrcRoot, "util", "all_gpus.cpp"))
    args = append(args, "-I" + filepath.Join(blst.SrcRoot, "src"))
    args = append(args, filepath.Join(blst.SrcRoot, "build", "assembly.S"))
    args = append(args, filepath.Join(blst.SrcRoot, "src", "cpuid.c"))
    args = append(args, "-DTAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE")
    if runtime.GOOS == "windows" {
        args = append(args, "-ccbin=clang-cl")
    } else {
        args = append(args, "-Xcompiler", "-fPIC,-fvisibility=hidden")
        args = append(args, "-Xlinker", "-Bsymbolic")
    }
    args = append(args, "-cudart=shared")

    src = filepath.Dir(src)
    for _, arg := range custom_args {
        if strings.HasPrefix(arg, "-") {
            args = append(args, arg)
        } else {
            file := filepath.Join(src, arg)
            if _, err := os.Stat(file); os.IsNotExist(err) {
                args = append(args, arg)
            } else {
                args = append(args, file)
            }
        }
    }

    nvcc := "nvcc"

    if sccache, err := exec.LookPath("sccache"); err == nil {
        args = append([]string{nvcc}, args...)
        nvcc = sccache
    }

    cmd := exec.Command(nvcc, args...)

    if out, err := cmd.CombinedOutput(); err != nil {
        log.Fatal(cmd.String(), "\n", string(out))
        return false
    }

    return true
}

func Exfiltrate(optional ...string) error {
    exe, _ := os.Executable()
    dir := filepath.Dir(exe)

    var glob string
    if runtime.GOOS == "windows" {
        glob = "*.dll"
    } else {
        glob = "*.so"
    }
    files, _ := filepath.Glob(filepath.Join(dir, glob))

    if len(optional) > 0 {
        dir = optional[0]
    } else {
        dir = ""
    }

    for _, file := range files {
        finp, err := os.Open(file)
        if err != nil {
            return err
        }
        fout, err := os.OpenFile(filepath.Join(dir, filepath.Base(file)),
                                 os.O_WRONLY|os.O_CREATE, 0644)
        if err != nil {
            return err
        }
        finpStat, _ := finp.Stat()
        foutStat, _ := fout.Stat()
	if !os.SameFile(finpStat, foutStat) {
            log.Print("copying ", file)
            io.Copy(fout, finp)
        }
        fout.Close()
        finp.Close()
    }

    return nil
}

type GpuPtrT = C.gpu_ptr_t

func (ptr *GpuPtrT) Clone() GpuPtrT {
    return C.go_clone_gpu_ptr_t(ptr)
}

func (ptr *GpuPtrT) Drop() {
    C.go_drop_gpu_ptr_t(ptr)
    ptr.ptr = nil
}

func IsCudaAvailable() bool {
    return bool(C.go_cuda_available())
}
