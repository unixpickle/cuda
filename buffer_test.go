package cuda

import (
	"reflect"
	"testing"
)

func TestBufferIO(t *testing.T) {
	ctx, a := setupTest(t)
	ctx.Run(func() error {
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
		if err := WriteBuffer(buf1, floatSize*3, []float32{1, 2}); err != nil {
			t.Error(err)
			return nil
		}
		if err := WriteBuffer(buf1, floatSize*8, []float32{3, 4, 5, 6, 7}); err != nil {
			t.Error(err)
			return nil
		}

		actual := make([]float32, 12)
		if err := ReadBuffer(actual, buf1, floatSize*1); err != nil {
			t.Error(err)
			return nil
		}
		expected := []float32{0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}

		if err := CopyBuffer(buf2, floatSize*5, buf1, floatSize*2); err != nil {
			t.Error(err)
			return nil
		}
		actual = make([]float32, 14)
		if err := ReadBuffer(actual, buf2, 0); err != nil {
			t.Error(err)
			return nil
		}
		expected = []float32{0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0}
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v but got %v", expected, actual)
		}
		return nil
	})
}
