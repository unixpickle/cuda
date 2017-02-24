package cublas

import (
	"testing"

	"github.com/unixpickle/cuda"
)

func TestSdgmm(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float32{1, 2, 3, 7, 4, 5, 6, 0},
		[]float32{0.5, 5, -2, 5, 3},
		[]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	<-ctx.Run(func() error {
		err := handle.Sdgmm(Left, 3, 2, buffers[0], 4, buffers[1], 2,
			buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float32, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float32{0.5, -4, 9, 0, 0, 2, -10, 18, 0, 0}

		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		err = handle.Sdgmm(Right, 3, 2, buffers[0], 3, buffers[1], 3,
			buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual = make([]float32, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float32{0.5, 1, 1.5, 35, 20, 25, -10, 18, 0, 0}

		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}
		return nil
	})
}

func TestDdgmm(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float64{1, 2, 3, 7, 4, 5, 6, 0},
		[]float64{0.5, 5, -2, 5, 3},
		[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	<-ctx.Run(func() error {
		err := handle.Ddgmm(Left, 3, 2, buffers[0], 4, buffers[1], 2,
			buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float64, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float64{0.5, -4, 9, 0, 0, 2, -10, 18, 0, 0}

		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		err = handle.Ddgmm(Right, 3, 2, buffers[0], 3, buffers[1], 3,
			buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual = make([]float64, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float64{0.5, 1, 1.5, 35, 20, 25, -10, 18, 0, 0}

		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}
		return nil
	})
}
