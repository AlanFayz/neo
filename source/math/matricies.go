package internalmath

import "fmt"

type Matrix struct {
	Values []float64
	Width  int
	Height int
}

func CreateMatrix(width, height int) *Matrix {
	matrix := Matrix{}
	matrix.Values = make([]float64, width*height)

	matrix.Width = width
	matrix.Height = height

	return &matrix
}

func (m *Matrix) Get(x, y int) float64 {
	return m.Values[x+y*m.Width]
}

func (m *Matrix) Set(x, y int, value float64) {
	m.Values[x+y*m.Width] = value
}

func (m *Matrix) Add(other *Matrix) *Matrix {
	if m.Height != other.Height && m.Width != other.Width {
		fmt.Println("dimensions of other matrix don't match")
		return nil
	}

	new := CreateMatrix(m.Width, m.Height)

	for y := 0; y < m.Height; y++ {
		for x := 0; x < m.Width; x++ {
			new.Set(x, y, m.Get(x, y)+other.Get(x, y))
		}
	}

	return new
}

func (m *Matrix) Sub(other *Matrix) *Matrix {
	if m.Height != other.Height && m.Width != other.Width {
		fmt.Println("dimensions of other matrix don't match")
		return nil
	}

	new := CreateMatrix(m.Width, m.Height)

	for y := 0; y < m.Height; y++ {
		for x := 0; x < m.Width; x++ {
			new.Set(x, y, m.Get(x, y)-other.Get(x, y))
		}
	}

	return new
}

func (m *Matrix) GetRow(row int) []float64 {
	result := make([]float64, 0)

	for i := 0; i < m.Width; i++ {
		result = append(result, m.Get(i, row))
	}

	return result
}

func (m *Matrix) Mul(other *Matrix) *Matrix {
	if m.Width != other.Height {
		fmt.Println("invalid dimensions for multiplication")
		return nil
	}

	rows := m.Height
	columns := other.Width

	new := CreateMatrix(rows, columns)

	for col := 0; col < columns; col++ {
		for row := 0; row < rows; row++ {

			sum := 0.0

			for k := 0; k < columns; k++ {
				sum += m.Get(row, k) * other.Get(k, col)
			}

			new.Set(row, col, sum)
		}
	}

	return new
}
