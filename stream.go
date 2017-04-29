package cuda

/*
#include <cuda.h>

const unsigned int streamNonBlockingFlag = CU_STREAM_NON_BLOCKING;
const CUstream nullStream = NULL;
*/
import "C"
import "unsafe"

// Synchronize waits for asynchronous operations to
// complete.
//
// This should be called in a Context.
func Synchronize() error {
	return newErrorDriver("cuCtxSynchronize", C.cuCtxSynchronize())
}

// A Stream manages a pipeline of CUDA operations.
// Streams can be employed to achieve parallelism.
type Stream struct {
	stream C.CUstream
	closed bool
}

// NewStream creates a new Stream.
//
// If nonBlocking is true, then this stream will be able
// to run concurrently with the default stream.
//
// This should be called in a Context.
func NewStream(nonBlocking bool) (*Stream, error) {
	res := &Stream{}
	status := C.cuStreamCreate(&res.stream, streamCreationFlags(nonBlocking))
	if err := newErrorDriver("cuStreamCreate", status); err != nil {
		return nil, err
	}
	return res, nil
}

// NewStreamPriority is like NewStream, but the resulting
// stream is assigned a certain priority.
//
// This should be called in a Context.
func NewStreamPriority(nonBlocking bool, priority int) (*Stream, error) {
	res := &Stream{}
	status := C.cuStreamCreateWithPriority(&res.stream, streamCreationFlags(nonBlocking),
		safeIntToC(priority))
	if err := newErrorDriver("cuStreamCreate", status); err != nil {
		return nil, err
	}
	return res, nil
}

// Synchronize waits for the stream's tasks to complete.
func (s *Stream) Synchronize() error {
	s.assertOpen()
	return newErrorDriver("cuStreamSynchronize", C.cuStreamSynchronize(s.stream))
}

// Close destroys the stream.
//
// This will return immediately, even if the stream is
// still doing work.
//
// A stream should not be used after it is closed.
//
// This should be called in a Context.
func (s *Stream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	return newErrorDriver("cuStreamDestroy", C.cuStreamDestroy(s.stream))
}

// Pointer returns the raw pointer value of the underlying
// stream object.
//
// If s is nil, then a NULL pointer is returned.
//
// This should be called in a Context.
func (s *Stream) Pointer() unsafe.Pointer {
	if s == nil {
		return unsafe.Pointer(C.nullStream)
	}
	s.assertOpen()
	return unsafe.Pointer(s.stream)
}

func (s *Stream) cuStream() C.CUstream {
	if s == nil {
		return C.nullStream
	}
	s.assertOpen()
	return s.stream
}

func (s *Stream) assertOpen() {
	if s != nil && s.closed {
		panic("stream closed")
	}
}

func streamCreationFlags(nonBlocking bool) C.uint {
	if nonBlocking {
		return C.streamNonBlockingFlag
	} else {
		return 0
	}
}
