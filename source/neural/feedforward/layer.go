package feedforward

import (
	"fmt"
	"math/rand/v2"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type Layer struct {
	Activations      []float64
	WeightedInputs   []float64
	DerivativeValues []float64
	Weights          []float64
	Biases           []float64
	WeightGradients  []float64
	BiasGradients    []float64
	Inputs           []float64

	NeuronInCount       int
	NeuronOutCount      int
	GradientUpdateCount int
}

func CreateLayer(neuronsIn, neuronsOut int) Layer {
	layer := Layer{}

	layer.GradientUpdateCount = 1

	layer.NeuronInCount = neuronsIn
	layer.NeuronOutCount = neuronsOut

	layer.Activations = make([]float64, neuronsOut)
	layer.WeightedInputs = make([]float64, neuronsOut)
	layer.DerivativeValues = make([]float64, neuronsOut)

	layer.Biases = make([]float64, neuronsOut)
	layer.BiasGradients = make([]float64, neuronsOut)

	//need to have a set of weights to the out neurons for each neuron this layer has
	layer.Weights = make([]float64, neuronsIn*neuronsOut)
	layer.WeightGradients = make([]float64, neuronsIn*neuronsOut)

	//init weights and biases

	for i := 0; i < int(neuronsIn*neuronsOut); i++ {
		layer.Weights[i] = rand.NormFloat64() * 0.1
	}

	for i := 0; i < int(neuronsOut); i++ {
		layer.Biases[i] = rand.NormFloat64() * 0.1
	}

	return layer
}

func (l *Layer) FetchWeight(in, out int) float64 {
	return l.Weights[in+out*l.NeuronInCount]
}

// returns just a pointer to the current layer
func (l *Layer) FeedForward(incoming *Layer) *Layer {
	if incoming.NeuronOutCount != l.NeuronInCount {
		fmt.Println("outgoing count from incoming not the same as incoming count")
		return nil
	}

	l.Inputs = incoming.Activations

	for out := 0; out < int(l.NeuronOutCount); out++ {
		l.WeightedInputs[out] = 0
		for in := 0; in < int(l.NeuronInCount); in++ {
			weight := l.FetchWeight(in, out)

			l.WeightedInputs[out] += incoming.Activations[in]*weight + l.Biases[out]
		}

		l.Activations[out] = internalmath.Sigmoid(l.WeightedInputs[out])
	}

	return l
}

func (l *Layer) FeedInput(input []float64) *Layer {
	if len(input) != l.NeuronInCount {
		fmt.Println("outgoing count from incoming not the same as incoming count")
		return nil
	}

	l.Inputs = input

	for out := 0; out < int(l.NeuronOutCount); out++ {
		l.WeightedInputs[out] = 0

		for in := 0; in < int(l.NeuronInCount); in++ {
			weight := l.FetchWeight(in, out)

			l.WeightedInputs[out] += input[in]*weight + l.Biases[in]
		}

		l.Activations[out] = internalmath.Sigmoid(l.WeightedInputs[out])
	}

	return l
}

func (l *Layer) ClearGradients() {
	for i := 0; i < l.NeuronOutCount; i++ {
		for j := 0; j < l.NeuronInCount; j++ {
			l.WeightGradients[j+i*l.NeuronInCount] = 0
		}

		l.BiasGradients[i] = 0
	}

	l.GradientUpdateCount = 1
}

func (l *Layer) ApplyGradients(learningRate, count float64) {
	for i := 0; i < l.NeuronOutCount; i++ {
		for j := 0; j < l.NeuronInCount; j++ {
			index := j + i*l.NeuronInCount
			l.Weights[index] -= (l.WeightGradients[index] / float64(l.GradientUpdateCount)) * learningRate
		}

		l.Biases[i] -= (l.BiasGradients[i] / float64(l.GradientUpdateCount)) * learningRate
	}
}

func (l *Layer) UpdateGradients() {
	for out := 0; out < l.NeuronOutCount; out++ {
		derivative := l.DerivativeValues[out]
		for in := 0; in < l.NeuronInCount; in++ {
			index := in + out*l.NeuronInCount

			l.WeightGradients[index] += derivative * l.Inputs[in]
		}

		l.BiasGradients[out] = l.DerivativeValues[out]
	}
}

func (l *Layer) ComputeDerivativesBackPropagationOutputLayer(expectedOutput []float64) {
	l.GradientUpdateCount += 1

	//compute partial deriviatives for cost/a sigmoid/input
	for i := 0; i < len(l.DerivativeValues); i++ {
		costActivationDerivative := internalmath.CostDerivative(expectedOutput[i], l.Activations[i])
		activationInputDerivative := internalmath.SigmoidDerivative(l.WeightedInputs[i])

		l.DerivativeValues[i] = costActivationDerivative * activationInputDerivative
	}
}

func (l *Layer) ComputeDerivativesBackPropagation(front *Layer) {
	for out := 0; out < l.NeuronOutCount; out++ {
		value := 0.0
		for frontDerivativeIndex := 0; frontDerivativeIndex < len(front.DerivativeValues); frontDerivativeIndex++ {
			value += front.FetchWeight(out, frontDerivativeIndex) * front.DerivativeValues[frontDerivativeIndex]
		}

		value *= internalmath.SigmoidDerivative(l.WeightedInputs[out])
	}
}

func (l *Layer) ResetGradients() {
	for out := 0; out < l.NeuronOutCount; out++ {
		for in := 0; in < l.NeuronInCount; in++ {
			index := in + out*l.NeuronInCount
			l.WeightGradients[index] = 0
		}

		l.BiasGradients[out] = 0
	}
}
