# Go bridge for sppark

The goal is to make it possible to reuse modules targeting Rust with Go. The suggestion is to complement the CUDA source module with a Go bridge that would look something like the following. Assuming that `poc.cu` implements ~`exern "C" __attribute__((visibility("default")))`~ `SPPARK_FFI RustError::by_value cuda_func(void*)`:

```go
package poc_cu

// #include "cgo_sppark.h"
// WRAP_ERR(Error, cuda_func, void *ptr)
// {   toGoError(go_err, (*cuda_func.call)(ptr));   }
import "C"

import (
    sppark "github.com/supranational/sppark/go"
)

func init() {
    sppark.Load("poc.cu", "-arch=native")
}

func CudaFunc() {
    var err C.GoError
    C.go_cuda_func(&err, nil)
    if err.code != 0 {
        panic(err.message)
    }
}
```

In the presented case `sppark.Load()` attempts to load `poc.so`, a shared library with the name derived from the first argument to the method, that is expected to reside next to the **current** executable. If not found, the method will attempt to compile `poc.cu` with `nvcc` and retry to load it. There may be any number of wrappers implemented in the bridge module. And one needs a copy of [`cgo_sppark.h`](cgo_sppark.h) in the same directory. If so desired, the CUDA module and the Go bridge can be packaged into a Go module for the target application to `import`.

The nature of this Go module is such that if a user wants to compile the shared object prior the application being executed for the first time, it's on the user. Because that's where the actual CUDA code is. One way is to implement a test that won't even have to make any CUDA calls, it's sufficient to copy the generated shared object to a directory of your choice. Consider [`poc_test.go`](../poc/go/poc_test.go) as a template.
