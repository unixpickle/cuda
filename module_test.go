package cuda

import (
	"io/ioutil"
	"math"
	"testing"
	"unsafe"
)

func TestModule(t *testing.T) {
	ptx, err := ioutil.ReadFile("test_data/kernels.ptx")
	if err != nil {
		t.Fatal(err)
	}
	ctx, a := setupTest(t)
	<-ctx.Run(func() error {
		mod, err := NewModule(ctx, string(ptx))
		if err != nil {
			t.Error(err)
			return nil
		}

		doubleBuf, err := AllocBuffer(a, 8*1550)
		if err != nil {
			t.Error(err)
			return nil
		}
		floatBuf, err := AllocBuffer(a, 4*1550)
		if err != nil {
			t.Error(err)
			return nil
		}

		floatBuf.WithPtr(func(ptr unsafe.Pointer) {
			err = mod.Launch("my_fancy_kernel", 13, 1, 1, 128, 1, 1, 0, int(1550),
				float32(3.7), float64(2.5), int(-3), uint(5), doubleBuf, ptr)
		})

		if err != nil {
			t.Error(err)
			return nil
		}

		res32 := make([]float32, 1550)
		res64 := make([]float64, 1550)

		if err := ReadBuffer(res32, floatBuf); err != nil {
			t.Error(err)
			return nil
		}
		if err := ReadBuffer(res64, doubleBuf); err != nil {
			t.Error(err)
			return nil
		}

		expFloat := float32(-3 + 5 - 3.7)
		for i, a := range res32 {
			if math.Abs(float64(a-expFloat)) > 1e-4 {
				t.Errorf("entry %d: expected %v but got %v", i, expFloat, a)
				break
			}
		}

		for i, a := range res64 {
			x := float64(i) + 3.7 + 2.5 - 3
			if math.Abs(x-a) > 1e-5 {
				t.Errorf("entry %d: expected %v but got %v", i, x, a)
				break
			}
		}
		return nil
	})
}
