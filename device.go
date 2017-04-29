package cuda

/*
#include <cuda.h>
#include <assert.h>

CUdevice_attribute devattr_for_idx(int i) {
	CUdevice_attribute attrs[] = {
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
		CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
		CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
		CU_DEVICE_ATTRIBUTE_WARP_SIZE,
		CU_DEVICE_ATTRIBUTE_MAX_PITCH,
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
		CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK,
		CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
		CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
		CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
		CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
		CU_DEVICE_ATTRIBUTE_INTEGRATED,
		CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
		CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
		CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
		CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
		CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
		CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
		CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
		CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
		CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
		CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
		CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
		CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
		CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
		CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
		CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
		CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
		CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
		CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
		CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
		CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
		CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
		CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
		CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
		CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
		CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
	};
	assert(i >= 0 && i < sizeof(attrs)/sizeof(CUdevice_attribute));
	return attrs[i];
}
*/
import "C"

import "unsafe"

// DevAttr is a CUDA device attribute.
type DevAttr int

func (d DevAttr) cValue() C.CUdevice_attribute {
	if d < 0 || d > DevAttrCanUseHostPointerForRegisteredMem {
		panic("invalid DevAttr")
	}
	return C.devattr_for_idx(C.int(d))
}

// All supported device attributes.
const (
	DevAttrMaxThreadsPerBlock DevAttr = iota
	DevAttrMaxBlockDimX
	DevAttrMaxBlockDimY
	DevAttrMaxBlockDimZ
	DevAttrMaxGridDimX
	DevAttrMaxGridDimY
	DevAttrMaxGridDimZ
	DevAttrMaxSharedMemoryPerBlock
	DevAttrSharedMemoryPerBlock
	DevAttrTotalConstantMemory
	DevAttrWarpSize
	DevAttrMaxPitch
	DevAttrMaxRegistersPerBlock
	DevAttrRegistersPerBlock
	DevAttrClockRate
	DevAttrTextureAlignment
	DevAttrGPUOverlap
	DevAttrMultiprocessorCount
	DevAttrKernelExecTimeout
	DevAttrIntegrated
	DevAttrCanMapHostMemory
	DevAttrComputeMode
	DevAttrMaximumTexture1DWidth
	DevAttrMaximumTexture2DWidth
	DevAttrMaximumTexture2DHeight
	DevAttrMaximumTexture3DWidth
	DevAttrMaximumTexture3DHeight
	DevAttrMaximumTexture3DDepth
	DevAttrMaximumTexture2DLayeredWidth
	DevAttrMaximumTexture2DLayeredHeight
	DevAttrMaximumTexture2DLayeredLayers
	DevAttrMaximumTexture2DArrayWidth
	DevAttrMaximumTexture2DArrayHeight
	DevAttrMaximumTexture2DArrayNumslices
	DevAttrSurfaceAlignment
	DevAttrConcurrentKernels
	DevAttrECCEnabled
	DevAttrPCIBusID
	DevAttrPCIDeviceID
	DevAttrTCCDriver
	DevAttrMemoryClockRate
	DevAttrGlobalMemoryBusWidth
	DevAttrL2CacheSize
	DevAttrMaxThreadsPerMultiprocessor
	DevAttrAsyncEngineCount
	DevAttrUnifiedAddressing
	DevAttrMaximumTexture1DLayeredWidth
	DevAttrMaximumTexture1DLayeredLayers
	DevAttrCanTex2DGather
	DevAttrMaximumTexture2DGatherWidth
	DevAttrMaximumTexture2DGatherHeight
	DevAttrMaximumTexture3DWidthAlternate
	DevAttrMaximumTexture3DHeightAlternate
	DevAttrMaximumTexture3DDepthAlternate
	DevAttrPCIDomainID
	DevAttrTexturePitchAlignment
	DevAttrMaximumTexturecubemapWidth
	DevAttrMaximumTexturecubemapLayeredWidth
	DevAttrMaximumTexturecubemapLayeredLayers
	DevAttrMaximumSurface1DWidth
	DevAttrMaximumSurface2DWidth
	DevAttrMaximumSurface2DHeight
	DevAttrMaximumSurface3DWidth
	DevAttrMaximumSurface3DHeight
	DevAttrMaximumSurface3DDepth
	DevAttrMaximumSurface1DLayeredWidth
	DevAttrMaximumSurface1DLayeredLayers
	DevAttrMaximumSurface2DLayeredWidth
	DevAttrMaximumSurface2DLayeredHeight
	DevAttrMaximumSurface2DLayeredLayers
	DevAttrMaximumSurfacecubemapWidth
	DevAttrMaximumSurfacecubemapLayeredWidth
	DevAttrMaximumSurfacecubemapLayeredLayers
	DevAttrMaximumTexture1DLinearWidth
	DevAttrMaximumTexture2DLinearWidth
	DevAttrMaximumTexture2DLinearHeight
	DevAttrMaximumTexture2DLinearPitch
	DevAttrMaximumTexture2DMipmappedWidth
	DevAttrMaximumTexture2DMipmappedHeight
	DevAttrComputeCapabilityMajor
	DevAttrComputeCapabilityMinor
	DevAttrMaximumTexture1DMipmappedWidth
	DevAttrStreamPrioritiesSupported
	DevAttrGlobalL1CacheSupported
	DevAttrLocalL1CacheSupported
	DevAttrMaxSharedMemoryPerMultiprocessor
	DevAttrMaxRegistersPerMultiprocessor
	DevAttrManagedMemory
	DevAttrMultiGPUBoard
	DevAttrMultiGPUBoardGroupID
	DevAttrHostNativeAtomicSupported
	DevAttrSingleToDoublePrecisionPerfRatio
	DevAttrPageableMemoryAccess
	DevAttrConcurrentManagedAccess
	DevAttrComputePreemptionSupported
	DevAttrCanUseHostPointerForRegisteredMem
)

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
//
// This needn't be called from a Context.
func (d *Device) Name() (string, error) {
	res := (*C.char)(C.malloc(0x100))
	defer C.free(unsafe.Pointer(res))
	cuRes := C.cuDeviceGetName(res, 0xff, d.id)
	if err := newErrorDriver("cuDeviceGetName", cuRes); err != nil {
		return "", err
	}
	return C.GoString(res), nil
}

// Attr gets an attribute of the device.
//
// This needn't be called from a Context.
func (d *Device) Attr(attr DevAttr) (int, error) {
	var res C.int
	cuRes := C.cuDeviceGetAttribute(&res, attr.cValue(), d.id)
	if err := newErrorDriver("cuDeviceGetAttribute", cuRes); err != nil {
		return 0, err
	}
	return int(res), nil
}

// TotalMem gets the device's total memory.
//
// This needn't be called from a Context.
func (d *Device) TotalMem() (uint64, error) {
	var res C.size_t
	cuRes := C.cuDeviceTotalMem(&res, d.id)
	if err := newErrorDriver("cuDeviceTotalMem", cuRes); err != nil {
		return 0, err
	}
	return uint64(res), nil
}
