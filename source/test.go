package main

import (
	"fmt"	
	"github.com/DevAlgos/neo/source/algorithms"
)

func main() {
	fmt.Println("Hi")
	y := []float64{140, 155, 159, 179, 192, 200, 212, 215}
	x1 := []float64{60, 62, 67, 70, 71, 72, 75, 78}
	x2 := []float64{22, 25, 24, 20, 15, 14, 14, 11}
	comb := [][]float64{x1, x2}

	f := algorithms.DataInput{Y: y, X: comb}
	algorithms.LinearRegression(f)

}