// Package cuda provides bindings to the CUDA library.
//
// Building
//
// To use this package, you must tell Go how to link with
// CUDA.
// On Mac OS X, this might look like:
//
//     export CUDA_PATH="/Developer/NVIDIA/CUDA-8.0"
//     export DYLD_LIBRARY_PATH="$CUDA_PATH/lib":$DYLD_LIBRARY_PATH
//     export CPATH="$CUDA_PATH/include/"
//     export CGO_LDFLAGS="/usr/local/cuda/lib/libcuda.dylib $CUDA_PATH/lib/libcudart.dylib $CUDA_PATH/lib/libcublas.dylib $CUDA_PATH/lib/libcurand.dylib"
//
// On Linux, this might look like:
//
//     export CUDA_PATH=/usr/local/cuda
//     export CPATH="$CUDA_PATH/include/"
//     export CGO_LDFLAGS="$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcurand.so"
//     export LD_LIBRARY_PATH=$CUDA_PATH/lib64/
//
// Contexts
//
// Virtually every cuda API must be run from within a
// Context, which can be created like so:
//
//     devices, err := cuda.AllDevices()
//     if err != nil {
//         // Handle error.
//     }
//     if len(devices) == 0 {
//         // No devices found.
//     }
//     ctx, err := cuda.NewContext(devices[0], 10)
//     if err != nil {
//         // Handle error.
//     }
//
// To run code in a Context asynchronously, you can do the
// following:
//
//     ctx.Run(func() error {
//         // My code here.
//     })
//
// To run code synchronously, simply read from the
// resulting channel:
//
//     <-ctx.Run(func() error {
//         // My code here.
//     })
//
// You should never call ctx.Run() inside another call to
// ctx.Run(), for reasons that are documented on the
// Context.Run() method.
//
// Memory Management
//
// There are two ways to deal with memory: using Buffers,
// or using an Allocator directly with unsafe.Pointers.
// The Buffer API provides a high-level buffer interface
// with garbage collection and bounds checking.
// Most APIs use Buffers, including the APIs provided by
// sub-packages.
//
// No matter what, you will need an Allocator if you want
// to allocate memory.
// You can create an Allocator directly on top of CUDA:
//
//     allocator := cuda.GCAllocator(cuda.NativeAllocator(ctx), 0)
//
// Once you have an allocator, you can use it to allocate
// Buffer objects like so:
//
//     err := <-ctx.Run(func() error {
//         // Allocate 16 bytes.
//         buffer, err := cuda.AllocBuffer(allocator, 16)
//         if err != nil {
//             return err
//         }
//         // Use the buffer here...
//     })
//
// There are various functions to help you deal with
// buffers.
// The WriteBuffer() and ReadBuffer() functions allow you
// to copy Go slices to and from buffers.
// The Slice() function allows you to get a Buffer which
// points to a sub-region of a parent Buffer.
//
// Kernels
//
// To run kernels, you will use a Module.
// You can pass various Go primitives, unsafe.Pointers,
// and Buffers as kernel arguments.
//
// Sub-packages
//
// The cublas and curand sub-packages provide basic linear
// algebra routines and random number generators,
// respectively.
package cuda
