package cublas

import (
	"math"
	"testing"

	"github.com/unixpickle/cuda"
)

func TestSdot(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float32{1, 2, 3, 4, -2, -3, 5},
		[]float32{3, -1, 2, 3, -2, 0, 4, 2.5, 3.5}, []float32{0})

	<-ctx.Run(func() error {
		var res float32
		err := handle.Sdot(3, buffers[0], 1, buffers[1], 1, &res)
		if err != nil {
			t.Error(err)
			return nil
		}
		if math.Abs(float64(res)-7) > 1e-4 {
			t.Errorf("bad value: %f", res)
		}

		err = handle.Sdot(5, buffers[0], 1, buffers[1], 2, &res)
		if err != nil {
			t.Error(err)
			return nil
		}
		if math.Abs(float64(res)-10) > 1e-4 {
			t.Errorf("bad value: %f", res)
		}

		err = handle.SetPointerMode(Device)
		if err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Sdot(3, buffers[0], 3, buffers[1], 4, buffers[2])
		if err != nil {
			t.Error(err)
			return nil
		}

		resSlice := make([]float32, 1)
		err = cuda.ReadBuffer(resSlice, buffers[2])
		if err != nil {
			t.Error(err)
			return nil
		}

		if math.Abs(float64(resSlice[0])-12.5) > 1e-4 {
			t.Errorf("bad value: %f", resSlice[0])
		}

		return nil
	})
}

func TestDdot(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float64{1, 2, 3, 4, -2, -3, 5},
		[]float64{3, -1, 2, 3, -2, 0, 4, 2.5, 3.5}, []float64{0})

	<-ctx.Run(func() error {
		var res float64
		err := handle.Ddot(3, buffers[0], 1, buffers[1], 1, &res)
		if err != nil {
			t.Error(err)
			return nil
		}
		if math.Abs(res-7) > 1e-4 {
			t.Errorf("bad value: %f", res)
		}

		err = handle.Ddot(5, buffers[0], 1, buffers[1], 2, &res)
		if err != nil {
			t.Error(err)
			return nil
		}
		if math.Abs(res-10) > 1e-4 {
			t.Errorf("bad value: %f", res)
		}

		err = handle.SetPointerMode(Device)
		if err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Ddot(3, buffers[0], 3, buffers[1], 4, buffers[2])
		if err != nil {
			t.Error(err)
			return nil
		}

		resSlice := make([]float64, 1)
		err = cuda.ReadBuffer(resSlice, buffers[2])
		if err != nil {
			t.Error(err)
			return nil
		}

		if math.Abs(resSlice[0]-12.5) > 1e-4 {
			t.Errorf("bad value: %f", resSlice[0])
		}

		return nil
	})
}
