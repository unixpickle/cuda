package cuda

import "testing"

var testingContext *Context
var testingAllocator Allocator

func setupTest(t *testing.T) (*Context, Allocator) {
	if testingContext != nil {
		return testingContext, testingAllocator
	}
	devices, err := AllDevices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) == 0 {
		t.Fatal("no CUDA devices")
	}
	testingContext, err = NewContext(devices[0], 10)
	if err != nil {
		t.Fatal(err)
	}
	testingAllocator = GCAllocator(NativeAllocator(testingContext), 0)
	return testingContext, testingAllocator
}
