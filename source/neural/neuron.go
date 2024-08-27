package neural

import (
	"fmt"
	"math/rand/v2"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type SigmoidNeuron struct {
	Weights internalmath.Vector[float64]
	Bias    float64
}

func CreateNeuron(v *internalmath.Vector[float64], Bias float64) *SigmoidNeuron {
	return &SigmoidNeuron{Weights: *v.Copy(), Bias: Bias}
}

func CreateNeuronRandomized(size int) *SigmoidNeuron {
	Weights := internalmath.CreateVectorWithSize[float64](size)

	for i := 0; i < size; i++ {
		Weights.Data[i] = rand.Float64()
	}

	// want to keep Weights as values instead of pointers for less indirection
	return &SigmoidNeuron{Weights: *Weights, Bias: rand.Float64()}
}

// may add more in the future and allow user to specifcy an activation function with interface

func (s *SigmoidNeuron) ComputeSigmoid(input *internalmath.Vector[float64]) float64 {
	if input.Size() != s.Weights.Size() {
		fmt.Println("weight size: ", s.Weights.Size())
		fmt.Println("input size: ", input.Size())
		fmt.Println("input was not the same length as the Weights")
		return 0
	}

	return internalmath.Sigmoid(s.Weights.Dot(input))
}
