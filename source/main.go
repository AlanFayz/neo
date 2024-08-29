package main

import (
	"fmt"

	"github.com/DevAlgos/neo/source/algorithms"
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

	//kareem work

	network.FeedForward(testInput)

	fmt.Println("------- after training --------")
	fmt.Println(network.GetResult())

	fmt.Println("Hi")
	y := []float64{140, 155, 159, 179, 192, 200, 212, 215}
	x1 := []float64{60, 62, 67, 70, 71, 72, 75, 78}
	x2 := []float64{22, 25, 24, 20, 15, 14, 14, 11}
	comb := [][]float64{x1, x2}

	f := algorithms.DataInput{Y: y, X: comb}
	statData := algorithms.StatisticalData{}

	for _, indepVar := range f.X {
		statData.SquareSums = append(statData.SquareSums, algorithms.SquareSum(indepVar))
		statData.CrossYsums = append(statData.CrossYsums, algorithms.CrossSums(indepVar, f.Y))
	}
	statData.DotProduct = algorithms.DotProduct(f.X)
	statData.N = len(f.Y)
	fmt.Println(statData)

	// fin := []DataPoint{}
	// for index, y := range y{
	// 	t:=[]float64{float64(x1[index]), float64(x2[index])}
	// 	fin = append(fin, DataPoint{Y:float64(y),X:t})
	// }

}
