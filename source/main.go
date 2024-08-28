package main

import (
	"fmt"

	"github.com/DevAlgos/neo/source/neural/feedforward"
)

func main() {
	network := feedforward.CreateNeuralNetwork(2, 5, 4)

	testInput := make([]float64, 0)
	testInput = append(testInput, 1.0, 0.5)

	expectedInput := make([]float64, 0)
	expectedInput = append(expectedInput, 0.5, 0.0, 0.5, 1.0)

	network.FeedForward(testInput)

	fmt.Println("------- before training --------")
	fmt.Println(network.GetResult())

	network.ClearGradients()

	data := feedforward.DataGroup{}
	data.Expected = expectedInput
	data.Input = testInput
	data.LearningRate = 0.1

	for trainingCount := 0; trainingCount < 1000; trainingCount++ {
		network.Train(&data)
	}

	network.FeedForward(testInput)

	fmt.Println("------- after training --------")
	fmt.Println(network.GetResult())
}
