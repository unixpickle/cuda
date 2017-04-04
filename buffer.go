package cuda

/*
#include <cuda.h>
#include <cuda_runtime_api.h>
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
//
// This does not zero out the returned memory.
// To do that, you should use ClearBuffer().
func AllocBuffer(a Allocator, size uintptr) (Buffer, error) {
	ptr, err := a.Alloc(size)
	if err != nil {
		return nil, err
	}
	return WrapPointer(a, ptr, size), nil
}

// WrapPointer wraps a pointer in a Buffer.
// You must specify the Allocator from which the pointer
// originated and the size of the buffer.
//
// After calling this, you should not use the pointer
// outside of the buffer.
// The Buffer will automatically free the pointer.
func WrapPointer(a Allocator, ptr unsafe.Pointer, size uintptr) Buffer {
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

type slice struct {
	Buffer
	off  uintptr
	size uintptr
}

// Slice creates a Buffer which views some part of the
// contents of another Buffer.
// The start and end indexes are inclusive and exclusive,
// respectively.
func Slice(b Buffer, start, end uintptr) Buffer {
	if start > end || start > b.Size() || end > b.Size() {
		panic("index out of bounds")
	}
	return &slice{
		Buffer: b,
		off:    start,
		size:   end - start,
	}
}

func (s *slice) Size() uintptr {
	return s.size
}

func (s *slice) WithPtr(f func(p unsafe.Pointer)) {
	s.Buffer.WithPtr(func(p unsafe.Pointer) {
		f(unsafe.Pointer(uintptr(p) + s.off))
	})
}

// Overlap checks if two buffers overlap in memory.
func Overlap(b1, b2 Buffer) bool {
	var overlap bool
	b1.WithPtr(func(ptr1 unsafe.Pointer) {
		b2.WithPtr(func(ptr2 unsafe.Pointer) {
			overlap = uintptr(ptr1) < uintptr(ptr2)+uintptr(b2.Size()) &&
				uintptr(ptr2) < uintptr(ptr1)+uintptr(b1.Size())
		})
	})
	return overlap
}

// ClearBuffer writes zeros over the contents of a Buffer.
// It must be called from the correct Context.
func ClearBuffer(b Buffer) error {
	var res C.cudaError_t
	b.WithPtr(func(ptr unsafe.Pointer) {
		res = C.cudaMemset(ptr, 0, C.size_t(b.Size()))
	})
	return newErrorRuntime("cudaMemset", res)
}

// WriteBuffer writes the data from a slice into a Buffer.
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
func WriteBuffer(b Buffer, val interface{}) error {
	size := bytesForSlice(val)
	if size > b.Size() {
		size = b.Size()
	}
	if size == 0 {
		return nil
	}

	var res C.cudaError_t
	b.WithPtr(func(ptr unsafe.Pointer) {
		switch val := val.(type) {
		case []byte:
			res = C.cudaMemcpy(ptr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []float64:
			res = C.cudaMemcpy(ptr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []float32:
			res = C.cudaMemcpy(ptr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []int32:
			res = C.cudaMemcpy(ptr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		case []uint32:
			res = C.cudaMemcpy(ptr, unsafe.Pointer(&val[0]), C.size_t(size),
				C.cudaMemcpyHostToDevice)
		}
	})

	return newErrorRuntime("cudaMemcpy", res)
}

// ReadBuffer reads the data from a Buffer into a slice.
// This must be called from the correct Context.
//
// See WriteBuffer for details on supported slice types.
func ReadBuffer(val interface{}, b Buffer) error {
	size := bytesForSlice(val)
	if size > b.Size() {
		size = b.Size()
	}
	if size == 0 {
		return nil
	}

	var res C.cudaError_t
	b.WithPtr(func(ptr unsafe.Pointer) {
		switch val := val.(type) {
		case []byte:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), ptr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []float64:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), ptr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []float32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), ptr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []int32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), ptr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		case []uint32:
			res = C.cudaMemcpy(unsafe.Pointer(&val[0]), ptr, C.size_t(size),
				C.cudaMemcpyDeviceToHost)
		}
	})

	return newErrorRuntime("cudaMemcpy", res)
}

// CopyBuffer copies as many bytes as possible from src
// into dst.
//
// The two Buffers must not contain overlapping regions of
// memory.
func CopyBuffer(dst, src Buffer) error {
	size := dst.Size()
	if src.Size() < size {
		size = src.Size()
	}
	if size == 0 {
		return nil
	}

	var res C.cudaError_t
	dst.WithPtr(func(dstPtr unsafe.Pointer) {
		src.WithPtr(func(srcPtr unsafe.Pointer) {
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
