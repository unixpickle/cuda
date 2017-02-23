package cuda

/*
#import "cuda.h"
#import "cuda_runtime_api.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// A Buffer provides a high-level interface into an
// underlying CUDA buffer.
type Buffer interface {
	// Allocator is the Allocator from which the Buffer was
	// allocated.
	Allocator() Allocator

	// Size is the size of the Buffer.
	Size() uintptr

	// WithPtr runs f with the pointer contained inside the
	// Buffer.
	// During the call to f, it is guaranteed that the Buffer
	// wil not be garbage collected.
	// However, nothing should store a reference to ptr after
	// f has completed.
	WithPtr(f func(ptr unsafe.Pointer))
}

type buffer struct {
	alloc Allocator
	size  uintptr
	ptr   unsafe.Pointer
}

// AllocBuffer allocates a new Buffer.
//
// This must be called in the Allocator's Context.
func AllocBuffer(a Allocator, size uintptr) (Buffer, error) {
	ptr, err := a.Alloc(size)
	if err != nil {
		return nil, err
	}
	return WrapBuffer(a, ptr, size), nil
}

// WrapBuffer wraps a pointer in a Buffer.
// You must specify the Allocator from which the pointer
// originated and the size of the buffer.
//
// After calling this, you should not use the pointer
// outside of the buffer.
// The Buffer will automatically free the pointer.
func WrapBuffer(a Allocator, ptr unsafe.Pointer, size uintptr) Buffer {
	res := &buffer{alloc: a, size: size, ptr: ptr}
	runtime.SetFinalizer(res, func(obj *buffer) {
		allocator := obj.alloc
		go allocator.Context().Run(func() error {
			allocator.Free(obj.ptr, obj.size)
			return nil
		})
	})
	return res
}

func (b *buffer) Allocator() Allocator {
	return b.alloc
}

func (b *buffer) Size() uintptr {
	return b.size
}

func (b *buffer) WithPtr(f func(p unsafe.Pointer)) {
	f(b.ptr)
	runtime.KeepAlive(b)
}

// WriteBuffer writes the data in a slice to a Buffer,
// starting at the offset off in the Buffer.
// It must be called from the correct Context.
//
// Supported slice types are:
//
//     []byte
//     []float64
//     []float32
//     []int32
//     []uint32
//
// Similar to the copy() built-in, the maximum possible
// amount of data will be copied.
func WriteBuffer(b Buffer, off uintptr, val interface{}) error {
	size := bytesForSlice(val)
	if size == 0 || off >= b.Size() {
		return nil
	} else if size > b.Size()-off {
		size = b.Size() - off
	}

	var res C.cudaError_t
	b.WithPtr(func(ptr unsafe.Pointer) {
		offPtr := unsafe.Pointer(uintptr(ptr) + off)
		switch val := val.(type) {
		case []byte:
			res = C.cudaMemcpy(offPtr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []float64:
			res = C.cudaMemcpy(offPtr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []float32:
			res = C.cudaMemcpy(offPtr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []int32:
			res = C.cudaMemcpy(offPtr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []uint32:
			res = C.cudaMemcpy(offPtr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		}
	})

	return newErrorRuntime("cudaMemcpy", res)
}

// ReadBuffer reads the contents of a Buffer (starting at
// the offset off) into a slice.
// This must be called from the correct Context.
//
// See WriteBuffer for details on supported slice types.
func ReadBuffer(val interface{}, b Buffer, off uintptr) error {
	size := bytesForSlice(val)
	if size == 0 || off >= b.Size() {
		return nil
	} else if size > b.Size()-off {
		size = b.Size() - off
	}

	var res C.cudaError_t
	b.WithPtr(func(ptr unsafe.Pointer) {
		offPtr := unsafe.Pointer(uintptr(ptr) + off)
		switch val := val.(type) {
		case []byte:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), offPtr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []float64:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), offPtr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []float32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), offPtr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []int32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), offPtr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []uint32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), offPtr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		}
	})

	return newErrorRuntime("cudaMemcpy", res)
}

// CopyBuffer copies as many bytes as possible from src
// into dst, where both buffers are potentially offset.
func CopyBuffer(dst Buffer, dstOff uintptr, src Buffer, srcOff uintptr) error {
	if dstOff >= dst.Size() || srcOff >= src.Size() {
		return nil
	}
	size := dst.Size() - dstOff
	if src.Size()-srcOff < size {
		size = src.Size() - srcOff
	}

	var res C.cudaError_t
	dst.WithPtr(func(dstPtr unsafe.Pointer) {
		dstPtr = unsafe.Pointer(uintptr(dstPtr) + dstOff)
		src.WithPtr(func(srcPtr unsafe.Pointer) {
			srcPtr = unsafe.Pointer(uintptr(srcPtr) + srcOff)
			res = C.cudaMemcpy(dstPtr, srcPtr, C.size_t(size),
				C.cudaMemcpyDeviceToDevice)
		})
	})

	return newErrorRuntime("cudaMemcpy", res)
}

func bytesForSlice(val interface{}) uintptr {
	switch val := val.(type) {
	case []byte:
		return uintptr(len(val))
	case []float64:
		return 8 * uintptr(len(val))
	case []float32:
		return 4 * uintptr(len(val))
	case []int32:
		return 4 * uintptr(len(val))
	case []uint32:
		return 4 * uintptr(len(val))
	default:
		panic(fmt.Sprintf("unsupported type: %T", val))
	}
}
