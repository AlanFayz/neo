package main

import "github.com/DevAlgos/neo/source/math"

func main() {
	v1 := math.NewVector[float64](1.0, 2.0, 5.0, 1.0, 5.0, 9.0, 10.0)

	v1.Normalize().Mul(5.0)

	print(v1.ToString())
}
