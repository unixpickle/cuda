package cublas

/*
#include <cublas_v2.h>
*/
import "C"

import (
	"unsafe"

	"github.com/unixpickle/cuda"
)

// Sgemv performs single-precision matrix-vector
// multiplication.
//
// Matrices are stored in column-major order.
//
// The leading dimension lda may not be 0.
//
// The type of alpha and beta depends on the pointer mode.
// In Host mode, use float32 or *float32.
// In Device mode, user cuda.Buffer.
//
// This must be called inside the cuda.Context
func (h *Handle) Sgemv(trans Operation, m, n int, alpha interface{},
	matA cuda.Buffer, lda int, x cuda.Buffer, incx int, beta interface{},
	y cuda.Buffer, incy int) error {
	checkGemv(trans, m, n, matA.Size()/4, lda, x.Size()/4, incx, y.Size()/4, incy)

	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		x.WithPtr(func(xPtr unsafe.Pointer) {
			y.WithPtr(func(yPtr unsafe.Pointer) {
				if h.PointerMode() == Host {
					pointerizeInputs(&alpha, &beta)
					res = C.cublasSgemv(h.handle,
						trans.cValue(),
						safeIntToC(m), safeIntToC(n),
						(*C.float)(alpha.(*float32)),
						(*C.float)(aPtr), safeIntToC(lda),
						(*C.float)(xPtr), safeIntToC(incx),
						(*C.float)(beta.(*float32)),
						(*C.float)(yPtr), safeIntToC(incy))
				} else {
					alphaBeta32(alpha, beta, func(alpha, beta *C.float) {
						res = C.cublasSgemv(h.handle,
							trans.cValue(),
							safeIntToC(m), safeIntToC(n),
							alpha,
							(*C.float)(aPtr), safeIntToC(lda),
							(*C.float)(xPtr), safeIntToC(incx),
							beta,
							(*C.float)(yPtr), safeIntToC(incy))
					})
				}
			})
		})
	})

	return newError("cublasSgemv", res)
}

// Dgemv performs double-precision matrix-vector
// multiplication.
//
// Matrices are stored in column-major order.
//
// The leading dimension lda may not be 0.
//
// The type of alpha and beta depends on the pointer mode.
// In Host mode, use float64 or *float64.
// In Device mode, user cuda.Buffer.
//
// This must be called inside the cuda.Context
func (h *Handle) Dgemv(trans Operation, m, n int, alpha interface{},
	matA cuda.Buffer, lda int, x cuda.Buffer, incx int, beta interface{},
	y cuda.Buffer, incy int) error {
	checkGemv(trans, m, n, matA.Size()/8, lda, x.Size()/8, incx, y.Size()/8, incy)

	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		x.WithPtr(func(xPtr unsafe.Pointer) {
			y.WithPtr(func(yPtr unsafe.Pointer) {
				if h.PointerMode() == Host {
					pointerizeInputs(&alpha, &beta)
					res = C.cublasDgemv(h.handle,
						trans.cValue(),
						safeIntToC(m), safeIntToC(n),
						(*C.double)(alpha.(*float64)),
						(*C.double)(aPtr), safeIntToC(lda),
						(*C.double)(xPtr), safeIntToC(incx),
						(*C.double)(beta.(*float64)),
						(*C.double)(yPtr), safeIntToC(incy))
				} else {
					alphaBeta64(alpha, beta, func(alpha, beta *C.double) {
						res = C.cublasDgemv(h.handle,
							trans.cValue(),
							safeIntToC(m), safeIntToC(n),
							alpha,
							(*C.double)(aPtr), safeIntToC(lda),
							(*C.double)(xPtr), safeIntToC(incx),
							beta,
							(*C.double)(yPtr), safeIntToC(incy))
					})
				}
			})
		})
	})

	return newError("cublasDgemv", res)
}

func checkGemv(trans Operation, m, n int, matA uintptr, lda int, x uintptr,
	incx int, y uintptr, incy int) {
	if m < 0 || n < 0 {
		panic("dimension out of bounds")
	} else if incx <= 0 || incy <= 0 {
		panic("increment out of bounds")
	}
	if trans != NoTrans {
		m, n = n, m
	}
	checkMatrix(trans, lda, m, n, matA)
	if stridedSize(x, incx) < uintptr(n) {
		panic("index out of bounds")
	}
	if stridedSize(y, incy) < uintptr(m) {
		panic("index out of bounds")
	}
}
