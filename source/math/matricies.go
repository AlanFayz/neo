package internalmath

func MultiplyMatricies(matrix1 [][]float64, matrix2 [][]float64) [][]float64 {
	rows1 := len(matrix1)
	cols1 := len(matrix1[0])
	rows2 := len(matrix2)
	cols2 := len(matrix2[0])

	if cols1 != rows2 {
		panic("Matrices cannot be multiplied: incompatible dimensions")
	}

	result := make([][]float64, rows1)
	for i := range result {
		result[i] = make([]float64, cols2)
	}

	for i := 0; i < rows1; i++ {
		for j := 0; j < cols2; j++ {
			sum := 0.0
			for k := 0; k < cols1; k++ {
				sum += matrix1[i][k] * matrix2[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}
