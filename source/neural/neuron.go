package neural

import (
	"fmt"
	"math/rand/v2"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type SigmoidNeuron struct {
	Weights      internalmath.Vector[float64]
	Gradients    internalmath.Vector[float64]
	Bias         float64
	BiasGradient float64
}

func CreateNeuron(v *internalmath.Vector[float64], Bias float64) *SigmoidNeuron {
	return &SigmoidNeuron{Weights: *v.Copy(), Bias: Bias}
}

func CreateNeuronRandomized(size int) *SigmoidNeuron {
	weights := internalmath.CreateVectorWithSize[float64](size)
	gradients := internalmath.CreateVectorWithSize[float64](size)

	for i := 0; i < size; i++ {
		weights.Data[i] = rand.NormFloat64() * 0.1
	}

	// want to keep weights as values instead of pointers for less indirection
	return &SigmoidNeuron{Weights: *weights, Gradients: *gradients, Bias: rand.NormFloat64() * 0.1}
}

func (s *SigmoidNeuron) ComputeSigmoid(input *internalmath.Vector[float64]) float64 {
	if input.Size() != s.Weights.Size() {
		fmt.Println("weight size: ", s.Weights.Size())
		fmt.Println("input size: ", input.Size())
		fmt.Println("input was not the same length as the Weights")
		return 0
	}

	return internalmath.Sigmoid(s.Weights.Dot(input) + s.Bias)
}

func (s *SigmoidNeuron) Compute(input *internalmath.Vector[float64]) float64 {
	if input.Size() != s.Weights.Size() {
		fmt.Println("weight size: ", s.Weights.Size())
		fmt.Println("input size: ", input.Size())
		fmt.Println("input was not the same length as the Weights")
		return 0
	}

	return s.Weights.Dot(input) + s.Bias
}
