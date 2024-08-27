package main

import (
	internalmath "github.com/DevAlgos/neo/source/math"
	"github.com/DevAlgos/neo/source/neural"
)

func main() {
	testInputs := internalmath.CreateVector[float64](1.0, 2.0, 3.0, 5.0)
	network := neural.CreateNeuralNetwork(testInputs, 20, 50, 40, 5, 1)

	network.Compute()

	print(network.GetOutputs().ToString())
}
