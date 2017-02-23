// Package cublas provides bindings for the CUDA cuBLAS
// library.
package cublas

/*
#include <cublas_v2.h>
*/
import "C"

import (
	"runtime"

	"github.com/unixpickle/cuda"
)

// A Handle is used to make cuBLAS calls.
//
// A given Handle is bound to a specific cuda.Context.
type Handle struct {
	handle C.cublasHandle_t
	ctx    *cuda.Context

	ptrMode PointerMode
}

// NewHandle creates a new cuBLAS handle.
//
// This must be called inside the cuda.Context.
func NewHandle(ctx *cuda.Context) (*Handle, error) {
	res := &Handle{ctx: ctx, ptrMode: Host}
	err := newError("cublasCreate", C.cublasCreate(&res.handle))
	if err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(obj *Handle) {
		go obj.ctx.Run(func() error {
			C.cublasDestroy(obj.handle)
			return nil
		})
	})
	return res, nil
}

// PointerMode returns the current PointerMode.
//
// This must be called inside the cuda.Context.
func (h *Handle) PointerMode() PointerMode {
	return h.ptrMode
}

// SetPointerMode updates the current PointerMode.
//
// This must be called inside the cuda.Context.
func (h *Handle) SetPointerMode(p PointerMode) error {
	res := C.cublasSetPointerMode(h.handle, p.cPointerMode())
	if err := newError("cublasSetPointerMode", res); err != nil {
		return err
	}
	h.ptrMode = p
	return nil
}
