package cuda

import "testing"

func setupTest(t *testing.T) (*Context, Allocator) {
	ctx, err := NewContext(10)
	if err != nil {
		t.Fatal(err)
	}
	return ctx, GCAllocator(NewNativeAllocator(ctx), 0)
}
