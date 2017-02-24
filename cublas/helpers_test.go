package cublas

import (
	"errors"
	"testing"

	"github.com/unixpickle/cuda"
)

var testContext *cuda.Context
var testAllocator cuda.Allocator
var testHandle *Handle

func setupTest(t *testing.T, inBuffers ...interface{}) (*cuda.Context, *Handle, []cuda.Buffer) {
	if testContext == nil {
		devices, err := cuda.AllDevices()
		if err != nil {
			t.Fatal(err)
		}
		if len(devices) == 0 {
			t.Fatal("no CUDA devices")
		}
		testContext, err = cuda.NewContext(devices[0], -1)
		if err != nil {
			t.Fatal(err)
		}
		testAllocator = cuda.GCAllocator(cuda.NativeAllocator(testContext), 0)
	}
	if testHandle == nil {
		err := <-testContext.Run(func() (err error) {
			testHandle, err = NewHandle(testContext)
			return
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	outBufs := make([]cuda.Buffer, len(inBuffers))
	for i, x := range inBuffers {
		err := <-testContext.Run(func() (err error) {
			switch x := x.(type) {
			case []float32:
				outBufs[i], err = cuda.AllocBuffer(testAllocator, uintptr(len(x)*4))
			case []float64:
				outBufs[i], err = cuda.AllocBuffer(testAllocator, uintptr(len(x)*8))
			default:
				err = errors.New("unknown buffer type")
			}
			return
		})
		if err != nil {
			t.Fatalf("buffer %d: %s", i, err)
		}
		cuda.WriteBuffer(outBufs[i], x)
	}

	return testContext, testHandle, outBufs
}
