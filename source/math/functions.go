package internalmath

import (
	"fmt"
	"math"
	"math/rand"
)

func Sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func SigmoidDerivative(n float64) float64 {
	activation := Sigmoid(n)
	return activation * (1 - activation)
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

func Cost(expected, output float64) float64 {
	diff := output - expected
	return diff * diff
}

func CostDerivative(expected, output float64) float64 {
	return 2 * (output - expected)
}

func CostVector(expected *Vector[float64], output *Vector[float64]) float64 {
	if expected.Size() != output.Size() {
		fmt.Println("expected and output vectors must be of the same length")
		return 0
	}

	costSum := 0.0
	for i := 0; i < expected.Size(); i++ {
		costSum += Cost(expected.Data[i], output.Data[i])
	}

	return costSum / float64(expected.Size())
}

func CDF(probablities []float64) []float64 {
	cdf := make([]float64, len(probablities))
	cumulative := 0.0

	for i := range cdf {
		cumulative += probablities[i]
		cdf[i] = cumulative
	}

	return cdf
}

func FindIndexFromRight(val float64, cdf []float64) int {
	for i, cumulativeProb := range cdf {
		if cumulativeProb >= val {
			return i
		}
	}

	return len(cdf) - 1
}

func IndexChoice(probabilities []float64, count int) []int {
	values := make([]int, 0)

	cdf := CDF(probabilities)

	for i := 0; i < count; i++ {

		rnd := rand.Float64()
		sample := FindIndexFromRight(rnd, cdf)
		values = append(values, sample)
	}

	return values
}
