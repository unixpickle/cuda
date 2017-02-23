package cuda

import "testing"

func setupTest(t *testing.T) (*Context, Allocator) {
	devices, err := AllDevices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) == 0 {
		t.Fatal("no CUDA devices")
	}
	ctx, err := NewContext(devices[0], 10)
	if err != nil {
		t.Fatal(err)
	}
	return ctx, GCAllocator(NewNativeAllocator(ctx), 0)
}
