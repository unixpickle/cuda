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

func TestSscal(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float32{1, 2, 3, 4, -2, -3, 5}, []float32{0.25})
	<-ctx.Run(func() error {
		actions := []func() error{
			func() error {
				return handle.Sscal(4, float32(2), buffers[0], 2)
			},
			func() error {
				scaler := float32(2)
				return handle.Sscal(3, &scaler, buffers[0], 1)
			},
			func() error {
				if err := handle.SetPointerMode(Device); err != nil {
					t.Error(err)
					return nil
				}
				defer handle.SetPointerMode(Host)
				return handle.Sscal(7, buffers[1], buffers[0], 1)
			},
		}
		expected := [][]float32{
			{2, 2, 6, 4, -4, -3, 10},
			{4, 4, 12, 4, -4, -3, 10},
			{1, 1, 3, 1, -1, -0.75, 2.5},
		}
		for i, f := range actions {
			if err := f(); err != nil {
				t.Errorf("action %d: %s", i, err)
				return nil
			}
			actual := make([]float32, 7)
			if err := cuda.ReadBuffer(actual, buffers[0]); err != nil {
				t.Error(err)
				return nil
			}
			expected := expected[i]
			if maxDelta32(actual, expected) > 1e-4 {
				t.Errorf("action %d: expected %v but got %v", i, expected, actual)
			}
		}
		return nil
	})
}

func TestDscal(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float64{1, 2, 3, 4, -2, -3, 5}, []float64{0.25})
	<-ctx.Run(func() error {
		actions := []func() error{
			func() error {
				return handle.Dscal(4, float64(2), buffers[0], 2)
			},
			func() error {
				scaler := float64(2)
				return handle.Dscal(3, &scaler, buffers[0], 1)
			},
			func() error {
				if err := handle.SetPointerMode(Device); err != nil {
					t.Error(err)
					return nil
				}
				defer handle.SetPointerMode(Host)
				return handle.Dscal(7, buffers[1], buffers[0], 1)
			},
		}
		expected := [][]float64{
			{2, 2, 6, 4, -4, -3, 10},
			{4, 4, 12, 4, -4, -3, 10},
			{1, 1, 3, 1, -1, -0.75, 2.5},
		}
		for i, f := range actions {
			if err := f(); err != nil {
				t.Errorf("action %d: %s", i, err)
				return nil
			}
			actual := make([]float64, 7)
			if err := cuda.ReadBuffer(actual, buffers[0]); err != nil {
				t.Error(err)
				return nil
			}
			expected := expected[i]
			if maxDelta64(actual, expected) > 1e-4 {
				t.Errorf("action %d: expected %v but got %v", i, expected, actual)
			}
		}
		return nil
	})
}
