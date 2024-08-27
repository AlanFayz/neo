package internalmath

import (
	"fmt"
	"math"

	"golang.org/x/exp/constraints"
)

type Vector[T constraints.Float] struct {
	Data []T
}

func CreateVector[T constraints.Float](values ...T) *Vector[T] {

	vector := Vector[T]{}

	vector.Data = append(vector.Data, values...)
	return &vector
}

func CreateVectorWithSize[T constraints.Float](size int) *Vector[T] {
	return &Vector[T]{Data: make([]T, size)}
}

func (v *Vector[T]) ToString() string {
	return fmt.Sprintf("%v", v.Data)
}

func (v *Vector[T]) Size() int {
	return len(v.Data)
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
		fmt.Println("vector must have same length")
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
		print("magnitude cannot be zero when normalizing")
		return v
	}

	for i := range v.Data {
		v.Data[i] /= magnitude
	}

	return v
}

func (v *Vector[T]) Copy() *Vector[T] {
	newVector := Vector[T]{Data: make([]T, len(v.Data))}
	copy(newVector.Data, v.Data)
	return &newVector
}

func (v *Vector[T]) PushValue(data T) {
	v.Data = append(v.Data, data)
}
