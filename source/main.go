package main

import (
	internalmath "github.com/DevAlgos/neo/source/math"
	"github.com/DevAlgos/neo/source/neural"
)

func main() {
	network := neural.NeuralNetwork{}

	testInputs := internalmath.CreateVector[float64](1.0, 2.0, 3.0, 5.0)
	network.SetInputs(testInputs)

	layer := neural.Layer{}

	for i := 0; i < 5; i++ {
		layer.Neurons = append(layer.Neurons, *neural.CreateNeuron(internalmath.CreateVector[float64](1.0, 2.0, 3.0, 5.0), 5.0))
	}

	network.PushHiddenLayer(&layer)

	network.SetOutputCount(1)

	outNeuron := neural.CreateNeuronRandomized(5)
	network.SetOutputNeuron(outNeuron, 0)

	network.Randomize()
	network.Compute()

	print(network.GetOutputs().ToString())
}
