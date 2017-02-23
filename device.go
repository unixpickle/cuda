package cuda

/*
#import <cuda.h>
*/
import "C"

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
