package curand

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/approb"
	"github.com/unixpickle/cuda"
)

func TestGeneratorPseudo(t *testing.T) {
	devices, err := cuda.AllDevices()
	if err != nil {
		t.Fatal(err)
	} else if len(devices) == 0 {
		t.Fatal("no CUDA devices")
	}
	ctx, err := cuda.NewContext(devices[0], -1)
	if err != nil {
		t.Fatal(ctx)
	}
	allocator := cuda.GCAllocator(cuda.NativeAllocator(ctx), 0)
	err = <-ctx.Run(func() (resErr error) {
		defer func() {
			if err := recover(); err != nil {
				resErr = err.(error)
			}
		}()
		gen, err := NewGenerator(ctx, PseudoDefault)
		if err != nil {
			t.Error(err)
			return nil
		}
		samplers := testingSampleFuncs(allocator, gen)
		groundTruth := []func() float64{rand.NormFloat64, rand.NormFloat64,
			rand.Float64, rand.Float64}
		for i, sampler := range samplers {
			realSampler := groundTruth[i]
			corr := approb.Correlation(10000, 0.1, sampler, realSampler)
			if corr < 0.99 {
				t.Errorf("distribution %d was wrong", i)
			}
		}
		return nil
	})
	if err != nil {
		t.Error(err)
	}
}

func testingSampleFuncs(allocator cuda.Allocator, g *Generator) []func() float64 {
	buf, err := cuda.AllocBuffer(allocator, 16)
	if err != nil {
		panic(err)
	}
	getValue32 := func() float32 {
		res := make([]float32, 1)
		if err := cuda.ReadBuffer(res, buf); err != nil {
			panic(err)
		}
		return res[0]
	}
	getValue64 := func() float64 {
		res := make([]float64, 1)
		if err := cuda.ReadBuffer(res, buf); err != nil {
			panic(err)
		}
		return res[0]
	}
	return []func() float64{
		func() float64 {
			if err := g.Normal(buf, 0, 1); err != nil {
				panic(err)
			}
			return float64(getValue32())
		},
		func() float64 {
			if err := g.NormalDouble(buf, 0, 1); err != nil {
				panic(err)
			}
			return getValue64()
		},
		func() float64 {
			if err := g.Uniform(buf); err != nil {
				panic(err)
			}
			return float64(getValue32())
		},
		func() float64 {
			if err := g.UniformDouble(buf); err != nil {
				panic(err)
			}
			return getValue64()
		},
	}
}
