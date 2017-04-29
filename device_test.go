package cuda

import "testing"

func TestDeviceName(t *testing.T) {
	devices, err := AllDevices()
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range devices {
		name, err := d.Name()
		if err != nil {
			t.Errorf("device %d: %v", i, err)
		} else if len(name) == 0 {
			t.Errorf("device %d: empty name", i)
		}
	}
}

func TestDeviceAttr(t *testing.T) {
	devices, err := AllDevices()
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range devices {
		rate, err := d.Attr(DevAttrClockRate)
		if err != nil {
			t.Errorf("device %d: %v", i, err)
		} else if rate == 0 {
			t.Errorf("device %d: clock rate 0", i)
		}
	}
}

func TestDeviceTotalMem(t *testing.T) {
	devices, err := AllDevices()
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range devices {
		mem, err := d.TotalMem()
		if err != nil {
			t.Errorf("device %d: %v", i, err)
		} else if mem == 0 {
			t.Errorf("device %d: no memory", i)
		}
	}
}
