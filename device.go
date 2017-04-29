package cuda

/*
#include <cuda.h>
*/
import "C"

import "unsafe"

// Device contains a unique ID for a CUDA device.
type Device struct {
	id C.CUdevice
}

// AllDevices lists the available CUDA devices.
//
// This needn't be called from a Context.
func AllDevices() ([]*Device, error) {
	var count C.int
	cuRes := C.cuDeviceGetCount(&count)
	if err := newErrorDriver("cuDeviceGetCount", cuRes); err != nil {
		return nil, err
	}
	var res []*Device
	for i := C.int(0); i < count; i++ {
		var dev C.CUdevice
		cuRes = C.cuDeviceGet(&dev, i)
		if err := newErrorDriver("cuDeviceGet", cuRes); err != nil {
			return nil, err
		}
		res = append(res, &Device{id: dev})
	}
	return res, nil
}

// Name gets the device's identifier string.
func (d *Device) Name() (string, error) {
	res := (*C.char)(C.malloc(0x100))
	defer C.free(unsafe.Pointer(res))
	cuRes := C.cuDeviceGetName(res, 0xff, d.id)
	if err := newErrorDriver("cuDeviceGetName", cuRes); err != nil {
		return "", err
	}
	return C.GoString(res), nil
}

// TotalMem gets the device's total memory.
func (d *Device) TotalMem() (uint64, error) {
	var res C.size_t
	cuRes := C.cuDeviceTotalMem(&res, d.id)
	if err := newErrorDriver("cuDeviceTotalMem", cuRes); err != nil {
		return 0, err
	}
	return uint64(res), nil
}
