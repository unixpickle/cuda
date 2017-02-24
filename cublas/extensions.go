package cublas

/*
#include <cublas_v2.h>

const cublasSideMode_t goCublasLeft = CUBLAS_SIDE_LEFT;
const cublasSideMode_t goCublasRight = CUBLAS_SIDE_RIGHT;
*/
import "C"

import (
	"unsafe"

	"github.com/unixpickle/cuda"
)

// A SideMode specifies the side on which a matrix should
// be applied to another matrix.
type SideMode int

const (
	Left SideMode = iota
	Right
)

func (s SideMode) cValue() C.cublasSideMode_t {
	switch s {
	case Left:
		return C.goCublasLeft
	case Right:
		return C.goCublasRight
	default:
		panic("invalid SideMode")
	}
}

// Sdgmm multiplies a dense matrix by a diagonal matrix.
//
// The mode argument indicates on which side the diagonal
// matrix should be placed.
//
// This must be called inside the cuda.Context.
func (h *Handle) Sdgmm(mode SideMode, m, n int, matA cuda.Buffer, lda int,
	x cuda.Buffer, incx int, matC cuda.Buffer, ldc int) error {
	checkDgmm(mode, m, n, matA.Size()/4, lda, x.Size()/4, incx, matC.Size()/4, ldc)
	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		x.WithPtr(func(xPtr unsafe.Pointer) {
			matC.WithPtr(func(cPtr unsafe.Pointer) {
				res = C.cublasSdgmm(h.handle, mode.cValue(),
					safeIntToC(m), safeIntToC(n),
					(*C.float)(aPtr), safeIntToC(lda),
					(*C.float)(xPtr), safeIntToC(incx),
					(*C.float)(cPtr), safeIntToC(ldc))
			})
		})
	})
	return newError("cublasSdgmm", res)
}

// Ddgmm is like Sdgmm, but for double-precision.
//
// The mode argument indicates on which side the diagonal
// matrix should be placed.
//
// This must be called inside the cuda.Context.
func (h *Handle) Ddgmm(mode SideMode, m, n int, matA cuda.Buffer, lda int,
	x cuda.Buffer, incx int, matC cuda.Buffer, ldc int) error {
	checkDgmm(mode, m, n, matA.Size()/8, lda, x.Size()/8, incx, matC.Size()/8, ldc)
	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		x.WithPtr(func(xPtr unsafe.Pointer) {
			matC.WithPtr(func(cPtr unsafe.Pointer) {
				res = C.cublasDdgmm(h.handle, mode.cValue(),
					safeIntToC(m), safeIntToC(n),
					(*C.double)(aPtr), safeIntToC(lda),
					(*C.double)(xPtr), safeIntToC(incx),
					(*C.double)(cPtr), safeIntToC(ldc))
			})
		})
	})
	return newError("cublasDdgmm", res)
}

func checkDgmm(mode SideMode, m, n int, matA uintptr, lda int, x uintptr, incx int,
	matC uintptr, ldc int) {
	checkMatrix(NoTrans, lda, m, n, matA)
	checkMatrix(NoTrans, ldc, m, n, matC)

	neededX := uintptr(m)
	if mode == Right {
		neededX = uintptr(n)
	}
	if stridedSize(x, incx) < neededX {
		panic("index out of bounds")
	}
}
