package cuda

/*
#include <cuda_runtime_api.h>
#include <cuda.h>
*/
import "C"

import (
	"os"
	"runtime"
	"strconv"
	"unsafe"
)

const minGCThresh = 1 << 15

// An Allocator allocates and frees CUDA memory.
//
// In general, Allocators are bound to a Context, meaning
// that they should only be used from within that Context.
//
// Usually, you should prefer to use the Buffer type over
// a direct memory allocation, since Buffers take care of
// garbage collection for you.
//
// Allocators are not responsible for zeroing out returned
// memory.
type Allocator interface {
	// Get the Context in which all calls to this Allocator
	// should be made.
	//
	// Unlike Alloc and Free, this needn't be called from the
	// allocator's Context.
	Context() *Context

	// Allocate a chunk of CUDA memory.
	//
	// This should only be called from the Context.
	Alloc(size uintptr) (unsafe.Pointer, error)

	// Free a chunk of CUDA memory.
	//
	// The size passed to Free must be the same size that was
	// passed to Alloc().
	//
	// This should only be called from the Context.
	Free(ptr unsafe.Pointer, size uintptr)
}

// A nativeAllocator allocates directly using CUDA.
type nativeAllocator struct {
	ctx *Context
}

// NativeAllocator returns an Allocator that allocates
// directly from the CUDA APIs.
//
// The resulting Allocator should be wrapped with
// GCAllocator if you plan to use it with the Buffer API.
//
// This need not be called in a Context.
func NativeAllocator(ctx *Context) Allocator {
	return &nativeAllocator{ctx: ctx}
}

func (n *nativeAllocator) Context() *Context {
	return n.ctx
}

func (n *nativeAllocator) Alloc(size uintptr) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	return ptr, newErrorRuntime("cudaMalloc", C.cudaMalloc(&ptr, C.size_t(size)))
}

func (n *nativeAllocator) Free(ptr unsafe.Pointer, size uintptr) {
	C.cudaFree(ptr)
}

type gcAllocator struct {
	Allocator

	inUse  uintptr
	thresh uintptr
	ratio  float64
}

// GCAllocator wraps an Allocator in a new Allocator which
// automatically triggers garbage collections.
//
// The frac argument behaves similarly to the GOGC
// environment variable, except that GOGC is a percentage
// whereas frac is a ratio.
// Thus, a frac of 1.0 is equivalent to GOGC=100.
// If frac is 0, the value for GOGC is used.
//
// If you are implementing your own Allocator, you will
// likely want to wrap it with GCAllocator so that it
// works nicely with the Buffer API.
//
// This need not be called in a Context.
func GCAllocator(a Allocator, frac float64) Allocator {
	if frac == 0 {
		frac = 1
		if gogc := os.Getenv("GOGC"); gogc != "" {
			val, err := strconv.ParseFloat(gogc, 64)
			if err == nil {
				frac = val / 100
			}
		}
	}
	if frac <= 0 {
		panic("invalid frac argument")
	}

	return &gcAllocator{
		Allocator: a,
		inUse:     0,
		thresh:    minGCThresh,
		ratio:     frac + 1,
	}
}

func (g *gcAllocator) Alloc(size uintptr) (unsafe.Pointer, error) {
	res, err := g.Allocator.Alloc(size)
	if err != nil {
		return res, err
	}
	g.inUse += size
	if g.inUse > g.thresh {
		g.thresh = g.updatedThresh()
		runtime.GC()
	}
	return res, nil
}

func (g *gcAllocator) Free(ptr unsafe.Pointer, size uintptr) {
	g.Allocator.Free(ptr, size)
	g.inUse -= size
	if g.inUse < 0 {
		panic("more memory was freed than allocated")
	}
	t := g.updatedThresh()
	if t < g.thresh {
		g.thresh = t
	}
}

func (g *gcAllocator) updatedThresh() uintptr {
	newVal := float64(g.inUse) * g.ratio

	// Only matters on 32-bit systems.
	if newVal > float64(^uintptr(0)) {
		return ^uintptr(0)
	}

	res := uintptr(newVal)
	if res > minGCThresh {
		return res
	} else {
		return minGCThresh
	}
}
