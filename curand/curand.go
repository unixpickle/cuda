// Package curand binds the CUDA cuRAND API to Go.
package curand

/*
#include "curand.h"

curandRngType_t go_curand_rng_type(int idx) {
	curandRngType_t options[] = {
		CURAND_RNG_PSEUDO_DEFAULT,
		CURAND_RNG_PSEUDO_XORWOW,
		CURAND_RNG_PSEUDO_MRG32K3A,
		CURAND_RNG_PSEUDO_MTGP32,
		CURAND_RNG_PSEUDO_MT19937,
		CURAND_RNG_PSEUDO_PHILOX4_32_10,
		CURAND_RNG_QUASI_DEFAULT,
		CURAND_RNG_QUASI_SOBOL32,
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
		CURAND_RNG_QUASI_SOBOL64,
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
	};
	return options[idx];
}
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/unixpickle/cuda"
)

type Type int

// Available generations from cuRAND API.
const (
	PseudoDefault Type = iota
	PseudoXORWOW
	PseudoMRG32K3A
	PseudoMTGP32
	PseudoMT19937
	PseudoPHILOX43210
	QuasiDefault
	QuasiSobol32
	QuasiScrambledSobol32
	QuasiSobol64
	QuasiScrambledSobol64
)

// A Generator generates random numbers.
type Generator struct {
	gen C.curandGenerator_t
}

// NewGenerator creates a Generator for the given type.
//
// This must be called inside a cuda.Context.
func NewGenerator(t Type) (*Generator, error) {
	if t > QuasiScrambledSobol64 || t < 0 {
		panic("type out of bounds")
	}
	realType := C.go_curand_rng_type(C.int(t))
	res := &Generator{}
	code := C.curandCreateGenerator(&res.gen, realType)
	if err := newError("curandCreateGenerator", code); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(res, func(g *Generator) {
		C.curandDestroyGenerator(g.gen)
	})
	return res, nil
}

// Seed sets the seed for a pseudo-random generator.
func (g *Generator) Seed(seed int64) error {
	status := C.curandSetPseudoRandomGeneratorSeed(g.gen, C.ulonglong(seed))
	return newError("curandSetPseudoRandomGeneratorSeed", status)
}

// GenerateSeeds initializes the generator.
//
// Generally, you will not need to call GenerateSeeds
// yourself.
// This is because other functions (e.g. Uniform) do the
// initialization process automatically if needed.
//
// This must be called inside a cuda.Context.
func (g *Generator) GenerateSeeds() error {
	return newError("curandGenerateSeeds", C.curandGenerateSeeds(g.gen))
}

// Uniform generates uniformly-distributed 32-bit floats
// and saves them to the buffer.
//
// This must be called inside a cuda.Context.
func (g *Generator) Uniform(buf cuda.Buffer) error {
	var res error
	buf.WithPtr(func(ptr unsafe.Pointer) {
		status := C.curandGenerateUniform(g.gen, (*C.float)(ptr),
			C.size_t(buf.Size()/4))
		res = newError("curandGenerateUniform", status)
	})
	return res
}

// UniformDouble is like Uniform, but for 64-bit floats.
//
// This must be called inside a cuda.Context.
func (g *Generator) UniformDouble(buf cuda.Buffer) error {
	var res error
	buf.WithPtr(func(ptr unsafe.Pointer) {
		status := C.curandGenerateUniformDouble(g.gen, (*C.double)(ptr),
			C.size_t(buf.Size()/8))
		res = newError("curandGenerateUniformDouble", status)
	})
	return res
}

// Normal generates normally distributed floats.
//
// cuRAND may require that the number of floats is
// divisible by 2.
//
// This must be called inside a cuda.Context.
func (g *Generator) Normal(buf cuda.Buffer, mean, stddev float32) error {
	var res error
	buf.WithPtr(func(ptr unsafe.Pointer) {
		status := C.curandGenerateNormal(g.gen, (*C.float)(ptr),
			C.size_t(buf.Size()/4), C.float(mean), C.float(stddev))
		res = newError("curandGenerateNormal", status)
	})
	return res
}

// NormalDouble generates normally distributed doubles.
//
// This must be called inside a cuda.Context.
func (g *Generator) NormalDouble(buf cuda.Buffer, mean, stddev float64) error {
	var res error
	buf.WithPtr(func(ptr unsafe.Pointer) {
		status := C.curandGenerateNormalDouble(g.gen, (*C.double)(ptr),
			C.size_t(buf.Size()/8), C.double(mean), C.double(stddev))
		res = newError("curandGenerateNormalDouble", status)
	})
	return res
}
