package main

import (
	"fmt"
	// "github.com/DevAlgos/neo/source/algorithms"
	internalmath "github.com/DevAlgos/neo/source/math"


	// "gonum.org/v1/plot"
	// "gonum.org/v1/plot/plotter"
	// "gonum.org/v1/plot/plotutil"
	// "gonum.org/v1/plot/vg"
)

// randomPoints returns some random x, y points.
// func Points(xs [][]float64,ys[]float64) plotter.XYs {
// 	pts := make(plotter.XYs, len(xs[0])*2)
// 	for ind,set := range xs{
// 		for index,y := range ys{
// 			pts[index+len(xs[0])*ind].X = set[index]
// 			pts[index+len(xs[0])*ind].Y = y
// 		}
// 	}
// 	return pts
// }


// func LineGraphPoints(model algorithms.RegressionModel) plotter.XYs {
// 	pts := make(plotter.XYs, 50)

// 	for i := range 50{
// 		pts[i].X = float64(i)
// 		arr := []float64{float64(i),0}
// 		pts[i].Y = model.Predict(arr)
// 	}
// 	return pts

// }

// func LineGraphPoints2(model algorithms.RegressionModel) plotter.XYs {
// 	pts := make(plotter.XYs, 50)

// 	for i := range 50{
// 		pts[i].X = float64(i)
// 		arr := []float64{0,float64(i)}
// 		pts[i].Y = model.Predict(arr)
// 	}
// 	return pts

// }

func Transpose(matrix [][]float64) [][]float64 {
	final := [][]float64{}
	curr := []float64{}
	for i := range matrix[0]{
		for j := range matrix{
			curr = append(curr,matrix[j][i])
		}

		final = append(final,curr)
		curr = nil

	}
	return final

}
// func multiplyMatrices(A, B [][]float64) [][]float64 {
// 	rowsA, colsA := len(A), len(A[0])
// 	rowsB, colsB := len(B), len(B[0])

// 	// Check if matrices can be multiplied
// 	if colsA != rowsB {
// 		fmt.Println("Matrix multiplication is not valid.")
// 		return nil
// 	}

// 	// Initialize result matrix C
// 	C := make([][]float64, rowsA)
// 	for i := range C {
// 		C[i] = make([]float64, colsB)
// 	}

// 	// Perform multiplication
// 	for i := 0; i < rowsA; i++ {
// 		for j := 0; j < colsB; j++ {
// 			for k := 0; k < colsA; k++ {
// 				C[i][j] += A[i][k] * B[k][j]
// 			}
// 		}
// 	}

// 	return C
// }


func createSliceWithOnes(length int) []float64 {
        ones := make([]float64, length)
        for i := 0; i < length; i++ {
                ones[i] = 1
        }
        return ones
}

func inverse(matrix [][]float64) ([][]float64, ) {
	n := len(matrix)

	// Create an identity matrix of the same size
	inverse := make([][]float64, n)
	for i := range inverse {
		inverse[i] = make([]float64, n)
		inverse[i][i] = 1
	}

	// Apply Gaussian elimination
	for i := 0; i < n; i++ {
		// Find the pivot element
		pivot := matrix[i][i]
		if pivot == 0 {
			return nil
		}

		// Scale the pivot row
		for j := 0; j < n; j++ {
			matrix[i][j] /= pivot
			inverse[i][j] /= pivot
		}

		// Eliminate the column elements
		for k := 0; k < n; k++ {
			if k != i {
				factor := matrix[k][i]
				for j := 0; j < n; j++ {
					matrix[k][j] -= factor * matrix[i][j]
					inverse[k][j] -= factor * inverse[i][j]
				}
			}
		}
	}

	return inverse
}

func SlowArrayDown(inp []float64) [][]float64{
	final := [][]float64{}
	curr := []float64{}
	for _,i := range inp{
		final = append(final,append(curr,i))

	}
	return final
}

func combineArrays(inputs ...[]float64) []float64 {
        result := []float64{}
        for _, arr := range inputs {
                result = append(result, arr...)
        }
        return result
}

func main() {
	// fmt.Println("Hi")
	// 215 78 11
	// y := []float64{140, 155, 159, 179, 192, 200, 212,215}
	// x1 := []float64{60, 62, 67, 70, 71, 72, 75, 78}
	// x2 := []float64{22, 25, 24, 20, 15, 14, 14,11}
	x1 := []float64{55,46,30,35,59,61,74,38,27,51,53,41,37,24,42,50,58,60,62,68,70,79,63,39,49}
	x2 := []float64{50,24,46,48,58,60,65,42,42,50,38,30,31,34,30,48,61,71,62,38,41,66,31,42,41}
	x3 := []float64{0,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,0,1}
	x4 := []float64{2.1,2.8,3.3,4.5,2,5.1,5.5,3.2,3.1,2.4,2.2,2.1,1.9,3.1,3,4.2,4.6,5.3,7.2,7.8,7,6.2,4.1,3.5,2.1}
	y := []float64{68,77,96,80,43,44,26,88,75,57,56,88,88,102,88,70,52,43,46,56,59,26,52,83,75}
	
	pad := createSliceWithOnes(len(y))
	// comb := [][]float64{x1, x2,x3,x4}
	t := [][]float64{pad,x1, x2,x3,x4}
	xt := Transpose(t)


	// fp := inverse(internalmath.MultiplyMatricies(t,xt))
	for _,x := range internalmath.MultiplyMatricies(SlowArrayDown(y),xt){
		fmt.Println(x)
	}

	
	
	// f := algorithms.DataInput{Y: y, X: comb}
	// model := algorithms.LinearRegression(f)

	// fmt.Println(model.Predict(t))

}

	// p := plot.New()

	// p.Title.Text = "Plotutil example"
	// p.X.Label.Text = "X"
	// p.Y.Label.Text = "Y"

	// scatter, err := plotter.NewScatter( Points(comb,y))
	// if err != nil {
	// 	panic(err)
	// }
	// err = plotutil.AddLinePoints(p,LineGraphPoints(model),LineGraphPoints2(model))
	// p.Add(scatter)
	// if err != nil {
	// 	panic(err)
	// }

	// // Save the plot to a PNG file.
	// if err := p.Save(10*vg.Inch, 10*vg.Inch, "points.png"); err != nil {
	// 	panic(err)
	// }



	// fmt.Println(model)
	// var inp1,inp2 int
	// fmt.Print("Input 1: ")
	// fmt.Scanln(&inp1)
	// fmt.Print("Input 2: ")
	// fmt.Scanln(&inp2)

	// inputs := []float64{float64(inp1),float64(inp2)}
	// fmt.Println(model.Predict(inputs))



