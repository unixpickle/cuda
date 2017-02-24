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
		runTestActions32(t, actions, expected, buffers[0])
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
		runTestActions64(t, actions, expected, buffers[0])
		return nil
	})
}

func TestSaxpy(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float32{1, 2, 3, 4, -2, -3, 5},
		[]float32{1, 0, -1, 0, 1, 2, -2, 3, 0}, []float32{3})
	<-ctx.Run(func() error {
		actions := []func() error{
			func() error {
				return handle.Saxpy(5, float32(2), buffers[0], 1, buffers[1], 2)
			},
			func() error {
				scaler := float32(-2)
				return handle.Saxpy(3, &scaler, buffers[0], 2, buffers[1], 3)
			},
			func() error {
				if err := handle.SetPointerMode(Device); err != nil {
					t.Error(err)
					return nil
				}
				defer handle.SetPointerMode(Host)
				return handle.Saxpy(2, buffers[2], buffers[0], 1, buffers[1], 1)
			},
		}
		expected := [][]float32{
			{3, 0, 3, 0, 7, 2, 6, 3, -4},
			{1, 0, 3, -6, 7, 2, 10, 3, -4},
			{4, 6, 3, -6, 7, 2, 10, 3, -4},
		}
		runTestActions32(t, actions, expected, buffers[1])
		return nil
	})
}

func TestDaxpy(t *testing.T) {
	ctx, handle, buffers := setupTest(t, []float64{1, 2, 3, 4, -2, -3, 5},
		[]float64{1, 0, -1, 0, 1, 2, -2, 3, 0}, []float64{3})
	<-ctx.Run(func() error {
		actions := []func() error{
			func() error {
				return handle.Daxpy(5, float64(2), buffers[0], 1, buffers[1], 2)
			},
			func() error {
				scaler := float64(-2)
				return handle.Daxpy(3, &scaler, buffers[0], 2, buffers[1], 3)
			},
			func() error {
				if err := handle.SetPointerMode(Device); err != nil {
					t.Error(err)
					return nil
				}
				defer handle.SetPointerMode(Host)
				return handle.Daxpy(2, buffers[2], buffers[0], 1, buffers[1], 1)
			},
		}
		expected := [][]float64{
			{3, 0, 3, 0, 7, 2, 6, 3, -4},
			{1, 0, 3, -6, 7, 2, 10, 3, -4},
			{4, 6, 3, -6, 7, 2, 10, 3, -4},
		}
		runTestActions64(t, actions, expected, buffers[1])
		return nil
	})
}

func runTestActions32(t *testing.T, fs []func() error, expected [][]float32, buf cuda.Buffer) {
	for i, f := range fs {
		if err := f(); err != nil {
			t.Errorf("action %d: %s", i, err)
			return
		}
		x := expected[i]
		actual := make([]float32, len(x))
		if err := cuda.ReadBuffer(actual, buf); err != nil {
			t.Error(err)
			return
		}
		if maxDelta32(actual, x) > 1e-4 {
			t.Errorf("action %d: expected %v but got %v", i, x, actual)
		}
	}
}

func runTestActions64(t *testing.T, fs []func() error, expected [][]float64, buf cuda.Buffer) {
	for i, f := range fs {
		if err := f(); err != nil {
			t.Errorf("action %d: %s", i, err)
			return
		}
		x := expected[i]
		actual := make([]float64, len(x))
		if err := cuda.ReadBuffer(actual, buf); err != nil {
			t.Error(err)
			return
		}
		if maxDelta64(actual, x) > 1e-4 {
			t.Errorf("action %d: expected %v but got %v", i, x, actual)
		}
	}
}
