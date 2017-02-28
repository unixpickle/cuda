package cublas

import (
	"testing"

	"github.com/unixpickle/cuda"
)

func TestSgemv(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]float32{3, 2, 1},
		[]float32{0, 0, 7, 6, 0, 0, 0, 0, 0, 0},
		[]float32{2.5},
		[]float32{3.1})
	<-ctx.Run(func() error {
		alpha := float32(2.5)
		err := handle.Sgemv(NoTrans, 3, 2, &alpha, buffers[0], 4, buffers[1], -2,
			float32(1), buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float32, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float32{40, 0, 7, 56, 0, 0, 60, 0, 0, 0}
		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := handle.SetPointerMode(Device); err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Sgemv(Trans, 3, 2, buffers[3], buffers[0], 5,
			buffers[1], -1, buffers[4], buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float32{159, 0, 7, 56, 0, 110, 60, 0, 0, 0}
		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		return nil
	})
}

func TestDgemv(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]float64{3, 2, 1},
		[]float64{0, 0, 7, 6, 0, 0, 0, 0, 0, 0},
		[]float64{2.5},
		[]float64{3.1})
	<-ctx.Run(func() error {
		alpha := float64(2.5)
		err := handle.Dgemv(NoTrans, 3, 2, &alpha, buffers[0], 4, buffers[1], -2,
			float64(1), buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float64, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float64{40, 0, 7, 56, 0, 0, 60, 0, 0, 0}
		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := handle.SetPointerMode(Device); err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Dgemv(Trans, 3, 2, buffers[3], buffers[0], 5,
			buffers[1], -1, buffers[4], buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float64{159, 0, 7, 56, 0, 110, 60, 0, 0, 0}
		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		return nil
	})
}
