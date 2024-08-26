package math

import (
	"fmt"
	"math"

	"golang.org/x/exp/constraints"
)

type Vector[T constraints.Float] struct {
	Data []T
}

func NewVector[T constraints.Float](values ...T) *Vector[T] {

	Vector := Vector[T]{}

	Vector.Data = append(Vector.Data, values...)
	return &Vector
}

func (v *Vector[T]) ToString() string {
	return fmt.Sprintf("%v", v.Data)
}

func (v *Vector[T]) Add(other *Vector[T]) *Vector[T] {
	min := math.Min(float64(len(other.Data)), float64(len(v.Data)))

	for i := 0; i < int(min); i++ {
		v.Data[i] += other.Data[i]
	}

	return v
}

func (v *Vector[T]) Sub(other *Vector[T]) *Vector[T] {

	min := math.Min(float64(len(other.Data)), float64(len(v.Data)))

	for i := 0; i < int(min); i++ {
		v.Data[i] -= other.Data[i]
	}

	return v
}

func (v *Vector[T]) Mul(other interface{}) *Vector[T] {
	switch other := other.(type) {
	case T:
		for i := range v.Data {
			v.Data[i] *= other
		}

	case *Vector[T]:
		min := math.Min(float64(len(other.Data)), float64(len(v.Data)))
		for i := 0; i < int(min); i++ {
			v.Data[i] *= other.Data[i]
		}
	}

	return v
}

func (v *Vector[T]) Div(other interface{}) *Vector[T] {
	switch other := other.(type) {
	case T:
		for i := range v.Data {
			v.Data[i] /= other
		}

	case *Vector[T]:
		min := math.Min(float64(len(other.Data)), float64(len(v.Data)))
		for i := 0; i < int(min); i++ {
			v.Data[i] /= other.Data[i]
		}
	}

	return v
}

func (v *Vector[T]) Dot(other *Vector[T]) T {
	if len(v.Data) != len(other.Data) {
		print("vector must have same length")
		return 0
	}

	var result T = 0

	for i := 0; i < len(v.Data); i++ {
		result += v.Data[i] * other.Data[i]
	}

	return result
}

func (v *Vector[T]) Magnitude() T {
	return T(math.Sqrt(float64(v.Dot(v))))
}

func (v *Vector[T]) Normalize() *Vector[T] {
	magnitude := v.Magnitude()
	if magnitude == 0 {
		return v
	}

	for i := range v.Data {
		v.Data[i] /= magnitude
	}

	return v
}
