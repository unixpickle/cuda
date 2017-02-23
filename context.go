package cuda

/*
#include "cuda.h"
*/
import "C"
import "runtime"

const defaultContextBuffer = 10

// A Context maintains a CUDA-dedicated thread.
// All CUDA code should be run by a Context.
type Context struct {
	msgs chan<- *contextMsg
}

// NewContext creates a new Context.
//
// The bufferSize is the maximum number of asynchronous
// calls that can be queued up at once.
// A bufferSize of -1 is replaced with a reasonable
// default.
func NewContext(bufferSize int) (*Context, error) {
	if bufferSize < -1 {
		panic("buffer size out of range")
	} else if bufferSize == -1 {
		bufferSize = defaultContextBuffer
	}
	msgs := make(chan *contextMsg, bufferSize)
	go contextLoop(msgs)
	res := &Context{msgs: msgs}
	err := <-res.Run(func() error {
		return NewErrorDriver("cuInit", C.cuInit(0))
	})
	if err != nil {
		close(msgs)
		return nil, err
	}
	runtime.SetFinalizer(res, func(obj interface{}) {
		close(obj.(*Context).msgs)
	})
	return res, nil
}

// Run runs f in the Context asynchronously and returns
// a channel that will be sent the result of f.
//
// If you are not interested in the result of f, you can
// simply ignore the returned channel.
//
// While f is running, no other function can run on the
// Context.
// This means that, to avoid deadlock, f should not use
// the Context in a blocking manner.
func (c *Context) Run(f func() error) <-chan error {
	ch := make(chan error, 1)
	msg := &contextMsg{
		f:        f,
		doneChan: ch,
	}
	c.msgs <- msg
	return ch
}

type contextMsg struct {
	f        func() error
	doneChan chan<- error
}

func contextLoop(msgs <-chan *contextMsg) {
	runtime.LockOSThread()
	for msg := range msgs {
		msg.doneChan <- msg.f()
		close(msg.doneChan)
	}
}
