package main

import (
	"fmt"

	internalmath "github.com/DevAlgos/neo/source/math"
	"github.com/DevAlgos/neo/source/neural"
)

func main() {
	testInputs := internalmath.CreateVector[float64](1.0, 2.0, 3.0, 5.0)
	network := neural.CreateNeuralNetwork(testInputs, 20, 50, 30, 10, 50, 20, 10)

	testInputs = internalmath.CreateVector[float64](5.4, 10000.0, 49.0, 60.4)

	network.Compute(testInputs)
	fmt.Println("------- before training --------")
	fmt.Println(network.GetOutputs().ToString())

	expectedOutput := internalmath.CreateVector[float64](1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.9, 1.0)

	dataGroup := neural.DataGroup{}
	dataGroup.Input = testInputs
	dataGroup.Expected = expectedOutput
	dataGroup.LearningRate = 0.05

	network.TrainNew(&dataGroup, 10000)

	network.Compute(testInputs)

	fmt.Println("------ after training -------")
	fmt.Println(network.GetOutputs().ToString())
}
