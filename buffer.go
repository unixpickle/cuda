package cuda

import (
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
