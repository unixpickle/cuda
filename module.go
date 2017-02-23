package cuda

/*
#include <string.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

const size_t ptrSize = sizeof(void *);
const size_t maxArgSize = 8;
const CUjit_option * nullJitOptions = NULL;
const void ** nullPtrPtr = NULL;
const CUstream nullStream = NULL;
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Synchronize waits for asynchronous operations to
// complete.
//
// This should be called in a Context.
func Synchronize() error {
	return newErrorDriver("cuCtxSynchronize", C.cuCtxSynchronize())
}

// A Module manages a set of compiled kernels.
type Module struct {
	module C.CUmodule
	cache  map[string]C.CUfunction
}

// NewModule creates a Module by compiling a chunk of PTX
// code.
//
// This should be called from within a Context.
//
// You can build PTX code using the nvcc compiler like so:
//
//     nvcc --gpu-architecture=compute_30 --gpu-code=compute_30 --ptx kernels.cu
//
// In the above example, you build "kernels.cu" to a PTX
// file called "kernels.ptx".
//
// The word size of the PTX should match the word size of
// the Go program.
// Depending on your use case, you may want to compile
// separate PTX files for 32-bit and 64-bit hosts.
func NewModule(ptx string) (*Module, error) {
	cstr := unsafe.Pointer(C.CString(ptx))
	defer C.free(cstr)

	var module C.CUmodule
	res := C.cuModuleLoadDataEx(&module, cstr, 0, C.nullJitOptions, C.nullPtrPtr)
	if err := newErrorDriver("cuModuleLoadDataEx", res); err != nil {
		return nil, err
	}

	m := &Module{module: module, cache: map[string]C.CUfunction{}}
	runtime.SetFinalizer(m, func(obj *Module) {
		C.cuModuleUnload(obj.module)
	})

	return m, nil
}

// Launch launches a kernel (which is referenced by name).
//
// This should be called from within the same Context that
// NewModule was called from.
//
// Currently, the following types may be used as kernel
// arguments:
//
//     uint
//     int
//     float32
//     float64
//     unsafe.Pointer
//     Buffer
//
// To wait for the launched kernel to complete, use
// the Synchronize() function.
func (m *Module) Launch(kernel string, gridX, gridY, gridZ, blockX, blockY, blockZ,
	sharedMem uint, args ...interface{}) error {
	res := cleanKernelArguments(args, nil, func(rawArgs []unsafe.Pointer) error {
		f, err := m.lookupKernel(kernel)
		if err != nil {
			return err
		}
		res := C.cuLaunchKernel(f, safeUintToC(gridX), safeUintToC(gridY),
			safeUintToC(gridZ), safeUintToC(blockX), safeUintToC(blockY),
			safeUintToC(blockZ), safeUintToC(sharedMem), C.nullStream,
			&rawArgs[0], C.nullPtrPtr)
		return newErrorDriver("cuLaunchKernel", res)
	})
	runtime.KeepAlive(m)
	return res
}

func (m *Module) lookupKernel(name string) (C.CUfunction, error) {
	if f, ok := m.cache[name]; ok {
		return f, nil
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var kernel C.CUfunction
	cuRes := C.cuModuleGetFunction(&kernel, m.module, cName)
	if err := newErrorDriver("cuModuleGetFunction", cuRes); err != nil {
		return kernel, err
	}
	m.cache[name] = kernel
	runtime.KeepAlive(m)
	return kernel, nil
}

func cleanKernelArguments(args []interface{}, newArgs []unsafe.Pointer,
	f func(args []unsafe.Pointer) error) error {
	if len(args) == 0 {
		return f(newArgs)
	}

	if buf, ok := args[0].(Buffer); ok {
		var res error
		buf.WithPtr(func(ptr unsafe.Pointer) {
			tempArgs := append([]interface{}{ptr}, args[1:]...)
			res = cleanKernelArguments(tempArgs, newArgs, f)
		})
		return res
	}

	valPtr := unsafe.Pointer(C.malloc(C.maxArgSize))
	defer C.free(valPtr)

	switch x := args[0].(type) {
	case uint:
		val := safeUintToC(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case int:
		val := safeIntToC(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case float32:
		val := C.float(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 4)
	case float64:
		val := C.double(x)
		C.memcpy(valPtr, unsafe.Pointer(&val), 8)
	case unsafe.Pointer:
		C.memcpy(valPtr, unsafe.Pointer(&x), C.ptrSize)
	}

	return cleanKernelArguments(args[1:], append(newArgs, valPtr), f)
}

func safeUintToC(x uint) C.uint {
	if x > uint(^C.uint(0)) {
		panic("uint value out of bounds")
	}
	return C.uint(x)
}

func safeIntToC(x int) C.int {
	if x > int(C.int(^C.uint(0)/2)) {
		panic("int value out of bounds")
	} else if x < int((-C.int(^C.uint(0)/2))-1) {
		panic("int value out of bounds")
	}
	return C.int(x)
}
