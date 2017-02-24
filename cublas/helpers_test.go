package cublas

import (
	"errors"
	"math"
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
			case []int32:
				outBufs[i], err = cuda.AllocBuffer(testAllocator, uintptr(len(x)*4))
			default:
				err = errors.New("unknown buffer type")
			}
			if err == nil {
				err = cuda.WriteBuffer(outBufs[i], x)
			}
			return
		})
		if err != nil {
			t.Fatalf("buffer %d: %s", i, err)
		}
	}

	return testContext, testHandle, outBufs
}

func maxDelta32(v1, v2 []float32) float32 {
	var delta float32
	for i, x := range v1 {
		y := v2[i]
		diff := float32(math.Abs(float64(x - y)))
		if diff > delta {
			delta = diff
		}
	}
	return delta
}

func maxDelta64(v1, v2 []float64) float64 {
	var delta float64
	for i, x := range v1 {
		y := v2[i]
		diff := math.Abs(x - y)
		if diff > delta {
			delta = diff
		}
	}
	return delta
}
