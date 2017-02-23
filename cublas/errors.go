package cublas

/*
#include <cublas_v2.h>

// Needed to check for NULL from Cgo.
const char * go_cublas_null_message = NULL;

const char * go_cublas_err(cublasStatus_t s) {
	switch (s) {
	case CUBLAS_STATUS_SUCCESS:
		return NULL;
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	default:
		return "unknown cuBLAS error";
	}
}
*/
import "C"

import "github.com/unixpickle/cuda"

// newError creates an Error from the result of a cuBLAS
// API call.
//
// If e is CUBLAS_STATUS_SUCCESS, nil is returned.
func newError(context string, e C.cublasStatus_t) error {
	cstr := C.go_cublas_err(e)
	if cstr == C.go_cublas_null_message {
		return nil
	}
	name := C.GoString(cstr)
	return &cuda.Error{
		Context: context,
		Name:    name,
		Message: name,
	}
}
