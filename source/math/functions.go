package internalmath

import (
	"fmt"
	"math"
)

func Sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func QuadraticCost(trainingInputCount uint64, expected *Vector[float64], output *Vector[float64]) float64 {
	if expected.Size() != output.Size() {
		fmt.Println("expected and output vectors must be of the same length")
		return 0
	}

	squaredSum := 0.0
	for i := 0; i < expected.Size(); i++ {
		diff := expected.Data[i] - output.Data[i]
		squaredSum += diff * diff
	}

	constant := 1.0 / (2.0 * float64(trainingInputCount))
	return constant * squaredSum
}
