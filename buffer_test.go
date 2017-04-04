package cuda

import (
	"reflect"
	"testing"
)

func TestBufferIO(t *testing.T) {
	ctx, a := setupTest(t)
	<-ctx.Run(func() error {
		const floatSize = 4
		buf1, err := AllocBuffer(a, floatSize*10)
		if err != nil {
			t.Error(err)
			return nil
		}
		buf2, err := AllocBuffer(a, floatSize*15)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := ClearBuffer(buf1); err != nil {
			t.Error(err)
			return nil
		}
		if err := ClearBuffer(buf2); err != nil {
			t.Error(err)
			return nil
		}
		if err := WriteBuffer(buf1, []float32{1, 2, 3, 0, 0, 4, 5}); err != nil {
			t.Error(err)
			return nil
		}
		actual := make([]float32, 8)
		if err := ReadBuffer(actual, buf1); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float32{1, 2, 3, 0, 0, 4, 5, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := CopyBuffer(buf2, buf1); err != nil {
			t.Error(err)
			return nil
		}

		actual = make([]float32, 15)
		if err := ReadBuffer(actual, buf2); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float32{1, 2, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		return nil
	})
}

func TestSlice(t *testing.T) {
	ctx, a := setupTest(t)
	<-ctx.Run(func() error {
		buf1, err := AllocBuffer(a, 32)
		if err != nil {
			t.Error(err)
			return nil
		}
		if err := ClearBuffer(buf1); err != nil {
			t.Error(err)
			return nil
		}
		if err := WriteBuffer(Slice(buf1, 8, 15), []byte{1, 2, 3, 4}); err != nil {
			t.Error(err)
			return nil
		}
		actual := make([]byte, 12)
		if err := ReadBuffer(actual, Slice(buf1, 4, 12)); err != nil {
			t.Error(err)
			return nil
		}
		expected := []byte{0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := CopyBuffer(Slice(buf1, 0, 4), Slice(buf1, 8, 16)); err != nil {
			t.Error(err)
			return nil
		}

		actual = make([]byte, 14)
		if err := ReadBuffer(actual, buf1); err != nil {
			t.Error(err)
			return nil
		}
		expected = []byte{1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if !Overlap(Slice(buf1, 0, 5), Slice(buf1, 3, 5)) {
			t.Error("should overlap")
		}
		if Overlap(Slice(buf1, 0, 5), Slice(buf1, 5, 10)) {
			t.Error("should not overlap")
		}

		return nil
	})
}
