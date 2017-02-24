package cuda

/*
#include <cuda.h>
#include <cuda_runtime_api.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/unixpickle/memalloc"
)

const (
	minAllocatorSize = 1 << 20
	maxAllocators    = 5

	allocAlignment = 32
	allocHeadroom  = 1 << 25
)

type bfcAllocator struct {
	a   []*memalloc.MemAllocator
	ctx *Context
}

// BFCAllocator creates an Allocator that uses memory
// coalescing and best-fitting to reduce memory
// fragmentation.
//
// You should wrap the returned allocator with GCAllocator
// if you plan to use the Buffer API.
//
// The maxSize argument specifies the maximum amount of
// memory to claim for the allocator.
// If it is 0, the allocator may claim nearly all of the
// available device memory.
//
// This should be called from a Context.
func BFCAllocator(ctx *Context, maxSize uintptr) (Allocator, error) {
	if maxSize == 0 {
		var free, total C.size_t
		err := newErrorRuntime("cudaGetMemInfo", C.cudaMemGetInfo(&free, &total))
		if err != nil {
			return nil, err
		}
		maxSize = uintptr(free)
		maxSize -= allocHeadroom
	}

	// No reason to reserve a misaligned amount of bytes.
	// Doing so would probably cause fragmentation, knowing
	// how bad cudaMalloc() is with fragmentation.
	maxSize = (maxSize / allocAlignment) * allocAlignment

	// The allocator size must fit in an int.
	for int(maxSize) < 0 || uintptr(int(maxSize)) != maxSize {
		maxSize >>= 1
	}

	var allocs []*memalloc.MemAllocator
	for len(allocs) < maxAllocators && maxSize >= minAllocatorSize {
		var region unsafe.Pointer
		err := newErrorRuntime("cudaMalloc", C.cudaMalloc(&region, C.size_t(maxSize)))
		if err != nil {
			maxSize >>= 1
			continue
		}
		allocs = append(allocs, &memalloc.MemAllocator{
			Start:     region,
			Size:      int(maxSize),
			Allocator: memalloc.NewBFC(int(maxSize), allocAlignment),
		})
	}
	if len(allocs) == 0 {
		return nil, errors.New("BFC init: not enough free memory")
	}

	res := &bfcAllocator{a: allocs, ctx: ctx}

	runtime.SetFinalizer(res, func(b *bfcAllocator) {
		go ctx.Run(func() error {
			for _, x := range b.a {
				C.cudaFree(x.Start)
			}
			return nil
		})
	})

	return res, nil
}

func (b *bfcAllocator) Context() *Context {
	return b.ctx
}

func (b *bfcAllocator) Alloc(size uintptr) (unsafe.Pointer, error) {
	if int(size) < 0 || uintptr(int(size)) != size {
		return nil, errors.New("BFC alloc: size must fit in int")
	}
	for _, x := range b.a {
		ptr, err := x.Alloc(int(size))
		if err == nil {
			return ptr, nil
		}
	}
	return nil, errors.New("BFC alloc: out of memory")
}

func (b *bfcAllocator) Free(ptr unsafe.Pointer, size uintptr) {
	for _, x := range b.a {
		if x.Contains(ptr) {
			x.Free(ptr)
			return
		}
	}
	panic("invalid pointer was freed")
}
