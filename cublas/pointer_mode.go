package cublas

/*
#include <cublas_v2.h>

const cublasPointerMode_t goCublasPointerModeHost = CUBLAS_POINTER_MODE_HOST;
const cublasPointerMode_t goCublasPointerModeDevice = CUBLAS_POINTER_MODE_DEVICE;
*/
import "C"

// PointerMode determines how BLAS APIs receive and return
// scaler values.
//
// There are two types of scaler values in the API: scaler
// inputs and scaler return values.
// The current pointer mode affects both types of values.
//
// If the pointer mode is Device, then all scaler inputs
// and outputs must be cuda.Buffer objects.
//
// If the pointer mode is Host, then all scaler inputs
// must be float32, float64, *float32, or *float64;
// all scaler outputs must be *float32 or *float64.
type PointerMode int

const (
	Host PointerMode = iota
	Device
)

func (p PointerMode) cPointerMode() C.cublasPointerMode_t {
	switch p {
	case Host:
		return C.goCublasPointerModeHost
	case Device:
		return C.goCublasPointerModeDevice
	default:
		panic("invalid PointerMode")
	}
}

// pointerizeInputs replaces float32 and float64 values
// with *float32 and *float64 values.
func pointerizeInputs(args ...*interface{}) {
	for _, x := range args {
		switch val := (*x).(type) {
		case float32:
			*x = &val
		case float64:
			*x = &val
		}
	}
}
