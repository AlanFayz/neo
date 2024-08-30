package main

import (
	"fmt"
	"time"

	internalmath "github.com/DevAlgos/neo/source/math"
	"github.com/DevAlgos/neo/source/neural/feedforward"
)

func main() {
	network := feedforward.CreateNeuralNetwork(2, 50, 50, 30, 20, 100, 4)

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
	data.LearningRate = 10.0

	current := time.Now()

	for trainingCount := 0; trainingCount < 100_000; trainingCount++ {
		network.Train(&data)
	}

	fmt.Println("------- training single threaded --------")
	fmt.Println(network.GetResult())

	fmt.Println("single threaded time took ", time.Since(current).Seconds(), " seconds")

	dataGroups := []feedforward.DataGroup{}

	for i := 0; i < 100_000; i++ {
		dataGroups = append(dataGroups, data)
	}

	network.ClearGradients()

	current = time.Now()

	network.TrainDataParallel(dataGroups, 10.0, 1)

	fmt.Println("------- training multi threaded --------")
	fmt.Println(network.GetResult())

	fmt.Println("single threaded time took ", time.Since(current).Seconds(), " seconds")

	matrix1 := [][]float64{}
	matrix2 := [][]float64{}
	matrix1 = append(matrix1, []float64{}, []float64{})
	matrix2 = append(matrix2, []float64{}, []float64{})

	matrix1[0] = append(matrix1[0], 1.0)
	matrix1[0] = append(matrix1[0], 4.0)

	matrix1[1] = append(matrix1[0], 1.0)
	matrix1[1] = append(matrix1[0], 4.0)

	matrix2[0] = append(matrix1[0], 1.0)
	matrix2[0] = append(matrix1[0], 4.0)

	matrix2[1] = append(matrix1[0], 1.0)
	matrix2[1] = append(matrix1[0], 4.0)

	fmt.Println(internalmath.MultiplyMatricies(matrix1, matrix2))
}
