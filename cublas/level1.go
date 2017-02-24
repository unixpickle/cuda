package cublas

import (
	"unsafe"

	"github.com/unixpickle/cuda"
)

/*
#include <cublas_v2.h>
*/
import "C"

// Sdot performs a single-precision dot product.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, it should be *float32.
// In the Device pointer mode, it should be a cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sdot(n int, x cuda.Buffer, incx int, y cuda.Buffer, incy int,
	result interface{}) error {
	if incx <= 0 || incy <= 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/4, incy) < uintptr(n) {
		panic("index out of bounds")
	}
	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				res = C.cublasSdot(h.handle, safeIntToC(n),
					(*C.float)(xPtr), safeIntToC(incx),
					(*C.float)(yPtr), safeIntToC(incy),
					(*C.float)(result.(*float32)))
			} else {
				result.(cuda.Buffer).WithPtr(func(outPtr unsafe.Pointer) {
					res = C.cublasSdot(h.handle, safeIntToC(n),
						(*C.float)(xPtr), safeIntToC(incx),
						(*C.float)(yPtr), safeIntToC(incy),
						(*C.float)(outPtr))
				})
			}
		})
	})
	return newError("cublasSdot", res)
}

// Ddot performs a double-precision dot product.
//
// The result argument's type depends on the pointer mode.
// In the Host pointer mode, it should be *float64.
// In the Device pointer mode, it should be a cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Ddot(n int, x cuda.Buffer, incx int, y cuda.Buffer, incy int,
	result interface{}) error {
	if incx <= 0 || incy <= 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/8, incy) < uintptr(n) {
		panic("index out of bounds")
	}
	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				res = C.cublasDdot(h.handle, safeIntToC(n),
					(*C.double)(xPtr), safeIntToC(incx),
					(*C.double)(yPtr), safeIntToC(incy),
					(*C.double)(result.(*float64)))
			} else {
				result.(cuda.Buffer).WithPtr(func(outPtr unsafe.Pointer) {
					res = C.cublasDdot(h.handle, safeIntToC(n),
						(*C.double)(xPtr), safeIntToC(incx),
						(*C.double)(yPtr), safeIntToC(incy),
						(*C.double)(outPtr))
				})
			}
		})
	})
	return newError("cublasDdot", res)
}

// Sscal scales a single-precision vector.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float32 or *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sscal(n int, alpha interface{}, x cuda.Buffer, incx int) error {
	if incx < 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(ptr unsafe.Pointer) {
		if h.PointerMode() == Host {
			pointerizeInputs(&alpha)
			res = C.cublasSscal(h.handle, safeIntToC(n), (*C.float)(alpha.(*float32)),
				(*C.float)(ptr), safeIntToC(incx))
		} else {
			alpha.(cuda.Buffer).WithPtr(func(alphaPtr unsafe.Pointer) {
				res = C.cublasSscal(h.handle, safeIntToC(n), (*C.float)(alphaPtr),
					(*C.float)(ptr), safeIntToC(incx))
			})
		}
	})

	return newError("cublasSscal", res)
}

// Dscal is like Sscal, but for double-precision.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float64 or *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Dscal(n int, alpha interface{}, x cuda.Buffer, incx int) error {
	if incx < 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(ptr unsafe.Pointer) {
		if h.PointerMode() == Host {
			pointerizeInputs(&alpha)
			res = C.cublasDscal(h.handle, safeIntToC(n), (*C.double)(alpha.(*float64)),
				(*C.double)(ptr), safeIntToC(incx))
		} else {
			alpha.(cuda.Buffer).WithPtr(func(alphaPtr unsafe.Pointer) {
				res = C.cublasDscal(h.handle, safeIntToC(n), (*C.double)(alphaPtr),
					(*C.double)(ptr), safeIntToC(incx))
			})
		}
	})

	return newError("cublasDscal", res)
}

// Saxpy computes single-precision "ax plus y".
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float32 or *float32.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Saxpy(n int, alpha interface{}, x cuda.Buffer, incx int, y cuda.Buffer,
	incy int) error {
	if incx < 0 || incy < 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/4, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/4, incy) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				pointerizeInputs(&alpha)
				res = C.cublasSaxpy(h.handle, safeIntToC(n), (*C.float)(alpha.(*float32)),
					(*C.float)(xPtr), safeIntToC(incx),
					(*C.float)(yPtr), safeIntToC(incy))
			} else {
				alpha.(cuda.Buffer).WithPtr(func(alphaPtr unsafe.Pointer) {
					res = C.cublasSaxpy(h.handle, safeIntToC(n), (*C.float)(alphaPtr),
						(*C.float)(xPtr), safeIntToC(incx),
						(*C.float)(yPtr), safeIntToC(incy))
				})
			}
		})
	})

	return newError("cublasSaxpy", res)
}

// Daxpy is like Saxpy, but for double-precision.
//
// The argument alpha's type depends on the pointer mode.
// In the Host pointer mode, use float64 or *float64.
// In the Device pointer mode, use cuda.Buffer.
//
// This must be called inside the cuda.Context.
func (h *Handle) Daxpy(n int, alpha interface{}, x cuda.Buffer, incx int, y cuda.Buffer,
	incy int) error {
	if incx < 0 || incy < 0 {
		panic("increment out of bounds")
	} else if n < 0 {
		panic("size out of bounds")
	} else if stridedSize(x.Size()/8, incx) < uintptr(n) {
		panic("index out of bounds")
	} else if stridedSize(y.Size()/8, incy) < uintptr(n) {
		panic("index out of bounds")
	}

	var res C.cublasStatus_t
	x.WithPtr(func(xPtr unsafe.Pointer) {
		y.WithPtr(func(yPtr unsafe.Pointer) {
			if h.PointerMode() == Host {
				pointerizeInputs(&alpha)
				res = C.cublasDaxpy(h.handle, safeIntToC(n), (*C.double)(alpha.(*float64)),
					(*C.double)(xPtr), safeIntToC(incx),
					(*C.double)(yPtr), safeIntToC(incy))
			} else {
				alpha.(cuda.Buffer).WithPtr(func(alphaPtr unsafe.Pointer) {
					res = C.cublasDaxpy(h.handle, safeIntToC(n), (*C.double)(alphaPtr),
						(*C.double)(xPtr), safeIntToC(incx),
						(*C.double)(yPtr), safeIntToC(incy))
				})
			}
		})
	})

	return newError("cublasDaxpy", res)
}

func stridedSize(totalCount uintptr, inc int) uintptr {
	// Do this in such a way that we never risk overflow.
	res := totalCount / uintptr(inc)
	if totalCount%uintptr(inc) != 0 {
		res++
	}
	return res
}
