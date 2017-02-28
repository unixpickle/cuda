package cublas

/*
#include <cublas_v2.h>

const cublasOperation_t goCublasOpN = CUBLAS_OP_N;
const cublasOperation_t goCublasOpT = CUBLAS_OP_T;
const cublasOperation_t goCublasOpC = CUBLAS_OP_C;
*/
import "C"

import (
	"unsafe"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/essentials"
)

// Operation specifies a matrix operation.
type Operation int

const (
	NoTrans Operation = iota
	Trans
	ConjTrans
)

func (o Operation) cValue() C.cublasOperation_t {
	switch o {
	case NoTrans:
		return C.goCublasOpN
	case Trans:
		return C.goCublasOpT
	case ConjTrans:
		return C.goCublasOpC
	default:
		panic("invalid Operation")
	}
}

// Sgemm performs single-precision matrix multiplication.
//
// Matrices are stored in column-major order.
//
// The leading dimensions lda, ldb, and ldc may not be 0.
//
// The type of alpha and beta depends on the pointer mode.
// In Host mode, use float32 or *float32.
// In Device mode, user cuda.Buffer.
//
// This must be called inside the cuda.Context
func (h *Handle) Sgemm(transA, transB Operation, m, n, k int, alpha interface{},
	matA cuda.Buffer, lda int, matB cuda.Buffer, ldb int, beta interface{},
	matC cuda.Buffer, ldc int) error {
	checkGemm(transA, transB, m, n, k,
		matA.Size()/4, lda,
		matB.Size()/4, ldb,
		matC.Size()/4, ldc)

	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		matB.WithPtr(func(bPtr unsafe.Pointer) {
			matC.WithPtr(func(cPtr unsafe.Pointer) {
				if h.PointerMode() == Host {
					pointerizeInputs(&alpha, &beta)
					res = C.cublasSgemm(h.handle,
						transA.cValue(), transB.cValue(),
						safeIntToC(m), safeIntToC(n), safeIntToC(k),
						(*C.float)(alpha.(*float32)),
						(*C.float)(aPtr), safeIntToC(lda),
						(*C.float)(bPtr), safeIntToC(ldb),
						(*C.float)(beta.(*float32)),
						(*C.float)(cPtr), safeIntToC(ldc))
				} else {
					alphaBeta32(alpha, beta, func(alpha, beta *C.float) {
						res = C.cublasSgemm(h.handle,
							transA.cValue(), transB.cValue(),
							safeIntToC(m), safeIntToC(n), safeIntToC(k),
							alpha,
							(*C.float)(aPtr), safeIntToC(lda),
							(*C.float)(bPtr), safeIntToC(ldb),
							beta,
							(*C.float)(cPtr), safeIntToC(ldc))
					})
				}
			})
		})
	})

	return newError("cublasSgemm", res)
}

// Dgemm is like Sgemm, but for double-precision.
//
// The type of alpha and beta depends on the pointer mode.
// In Host mode, use float64 or *float64.
// In Device mode, user cuda.Buffer.
//
// This must be called inside the cuda.Context
func (h *Handle) Dgemm(transA, transB Operation, m, n, k int, alpha interface{},
	matA cuda.Buffer, lda int, matB cuda.Buffer, ldb int, beta interface{},
	matC cuda.Buffer, ldc int) error {
	checkGemm(transA, transB, m, n, k,
		matA.Size()/8, lda,
		matB.Size()/8, ldb,
		matC.Size()/8, ldc)

	var res C.cublasStatus_t
	matA.WithPtr(func(aPtr unsafe.Pointer) {
		matB.WithPtr(func(bPtr unsafe.Pointer) {
			matC.WithPtr(func(cPtr unsafe.Pointer) {
				if h.PointerMode() == Host {
					pointerizeInputs(&alpha, &beta)
					res = C.cublasDgemm(h.handle,
						transA.cValue(), transB.cValue(),
						safeIntToC(m), safeIntToC(n), safeIntToC(k),
						(*C.double)(alpha.(*float64)),
						(*C.double)(aPtr), safeIntToC(lda),
						(*C.double)(bPtr), safeIntToC(ldb),
						(*C.double)(beta.(*float64)),
						(*C.double)(cPtr), safeIntToC(ldc))
				} else {
					alphaBeta64(alpha, beta, func(alpha, beta *C.double) {
						res = C.cublasDgemm(h.handle,
							transA.cValue(), transB.cValue(),
							safeIntToC(m), safeIntToC(n), safeIntToC(k),
							alpha,
							(*C.double)(aPtr), safeIntToC(lda),
							(*C.double)(bPtr), safeIntToC(ldb),
							beta,
							(*C.double)(cPtr), safeIntToC(ldc))
					})
				}
			})
		})
	})

	return newError("cublasDgemm", res)
}

func alphaBeta32(alpha, beta interface{}, f func(alpha, beta *C.float)) {
	b1 := alpha.(cuda.Buffer)
	b2 := beta.(cuda.Buffer)
	if b1.Size() < 4 || b2.Size() < 4 {
		panic("buffer underflow")
	}
	b1.WithPtr(func(ptr1 unsafe.Pointer) {
		b2.WithPtr(func(ptr2 unsafe.Pointer) {
			f((*C.float)(ptr1), (*C.float)(ptr2))
		})
	})
}

func alphaBeta64(alpha, beta interface{}, f func(alpha, beta *C.double)) {
	b1 := alpha.(cuda.Buffer)
	b2 := beta.(cuda.Buffer)
	if b1.Size() < 4 || b2.Size() < 4 {
		panic("buffer underflow")
	}
	b1.WithPtr(func(ptr1 unsafe.Pointer) {
		b2.WithPtr(func(ptr2 unsafe.Pointer) {
			f((*C.double)(ptr1), (*C.double)(ptr2))
		})
	})
}

func checkGemm(transA, transB Operation, m, n, k int, A uintptr, lda int, B uintptr,
	ldb int, C uintptr, ldc int) {
	if m < 0 || n < 0 || k < 0 {
		panic("dimension out of bounds")
	}
	checkMatrix(transA, lda, m, k, A)
	checkMatrix(transB, ldb, k, n, B)
	checkMatrix(NoTrans, ldc, m, n, C)
}

// checkMatrix ensures that op(A) fits in size elements,
// given that op(A) is a-by-b and has leading dimension
// lda.
func checkMatrix(op Operation, lda, a, b int, size uintptr) {
	if op == NoTrans {
		if lda < essentials.MaxInt(1, a) {
			panic("leading dimension out of bounds")
		}
		if size/uintptr(lda) < uintptr(b) {
			panic("index out of bounds")
		}
	} else {
		if lda < essentials.MaxInt(1, b) {
			panic("leading dimension out of bounds")
		}
		if size/uintptr(lda) < uintptr(a) {
			panic("index out of bounds")
		}
	}
}
