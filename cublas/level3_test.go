package cublas

import (
	"testing"

	"github.com/unixpickle/cuda"
)

func TestSgemm(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float32{1, 2, 3, 0, 4, 5, 6, 0},
		[]float32{-2, 0, 1, 2, -1, -1},
		[]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		[]float32{2.5},
		[]float32{3.1})
	<-ctx.Run(func() error {
		alpha := float32(2.5)
		err := handle.Sgemm(NoTrans, Trans, 3, 3, 2, &alpha, buffers[0], 4, buffers[1], 3,
			float32(0), buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float32, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float32{15, 15, 15, -10, -12.5, -15, -7.5, -7.5, -7.5, 0}
		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := handle.SetPointerMode(Device); err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Sgemm(Trans, NoTrans, 2, 2, 3, buffers[3], buffers[0], 4,
			buffers[1], 3, buffers[4], buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float32{49, 41.5, 15, -10, -12.5, -54, -30.750, -7.5, -7.5, 0}
		if maxDelta32(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		return nil
	})
}

func TestDgemm(t *testing.T) {
	ctx, handle, buffers := setupTest(t,
		[]float64{1, 2, 3, 0, 4, 5, 6, 0},
		[]float64{-2, 0, 1, 2, -1, -1},
		[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		[]float64{2.5},
		[]float64{3.1})
	<-ctx.Run(func() error {
		alpha := float64(2.5)
		err := handle.Dgemm(NoTrans, Trans, 3, 3, 2, &alpha, buffers[0], 4, buffers[1], 3,
			float64(0), buffers[2], 3)
		if err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float64, 10)
		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float64{15, 15, 15, -10, -12.5, -15, -7.5, -7.5, -7.5, 0}
		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := handle.SetPointerMode(Device); err != nil {
			t.Error(err)
			return nil
		}
		defer handle.SetPointerMode(Host)

		err = handle.Dgemm(Trans, NoTrans, 2, 2, 3, buffers[3], buffers[0], 4,
			buffers[1], 3, buffers[4], buffers[2], 5)
		if err != nil {
			t.Error(err)
			return nil
		}

		if err := cuda.ReadBuffer(actual, buffers[2]); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float64{49, 41.5, 15, -10, -12.5, -54, -30.750, -7.5, -7.5, 0}
		if maxDelta64(actual, expected) > 1e-4 {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		return nil
	})
}
