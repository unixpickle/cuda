package curand

/*
#include "curand.h"

// Needed to check for NULL from Cgo.
const char * goCurandNULLMessage = NULL;

const char * go_curand_err(curandStatus_t s) {
	switch (s) {
	case CURAND_STATUS_SUCCESS:
		return NULL;
	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	default:
		return "unknown cuRAND error";
	}
}
*/
import "C"

import "github.com/unixpickle/cuda"

// newError creates an Error from the result of a cuRAND
// API call.
//
// If e is CURAND_STATUS_SUCCESS, nil is returned.
func newError(context string, e C.curandStatus_t) error {
	msg := C.go_curand_err(e)
	if msg == C.goCurandNULLMessage {
		return nil
	}
	name := C.GoString(msg)
	return &cuda.Error{
		Context: context,
		Name:    name,
		Message: name,
	}
}
