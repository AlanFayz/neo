package main

import (
	// "fmt"
	"github.com/DevAlgos/neo/source/algorithms"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// randomPoints returns some random x, y points.
func Points(xs [][]float64,ys[]float64) plotter.XYs {
	pts := make(plotter.XYs, len(xs[0])*2)
	for ind,set := range xs{
		for index,y := range ys{
			pts[index+len(xs[0])*ind].X = set[index]
			pts[index+len(xs[0])*ind].Y = y
		}
	}
	return pts
}


func LineGraphPoints(model algorithms.RegressionModel) plotter.XYs {
	pts := make(plotter.XYs, 50)

	for i := range 50{
		pts[i].X = float64(i)
		arr := []float64{float64(i),0}
		pts[i].Y = model.Predict(arr)
	}
	return pts

}

func LineGraphPoints2(model algorithms.RegressionModel) plotter.XYs {
	pts := make(plotter.XYs, 50)

	for i := range 50{
		pts[i].X = float64(i)
		arr := []float64{0,float64(i)}
		pts[i].Y = model.Predict(arr)
	}
	return pts

}


func main() {
	// fmt.Println("Hi")
	// 215 78 11
	y := []float64{140, 155, 159, 179, 192, 200, 212,}
	x1 := []float64{60, 62, 67, 70, 71, 72, 75, }
	x2 := []float64{22, 25, 24, 20, 15, 14, 14,}
	comb := [][]float64{x1, x2}

	p := plot.New()

	
	f := algorithms.DataInput{Y: y, X: comb}
	model := algorithms.LinearRegression(f)

	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	scatter, err := plotter.NewScatter( Points(comb,y))
	if err != nil {
		panic(err)
	}
	err = plotutil.AddLinePoints(p,LineGraphPoints(model),LineGraphPoints2(model))
	p.Add(scatter)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(10*vg.Inch, 10*vg.Inch, "points.png"); err != nil {
		panic(err)
	}



	// fmt.Println(model)
	// var inp1,inp2 int
	// fmt.Print("Input 1: ")
	// fmt.Scanln(&inp1)
	// fmt.Print("Input 2: ")
	// fmt.Scanln(&inp2)

	// inputs := []float64{float64(inp1),float64(inp2)}
	// fmt.Println(model.Predict(inputs))

}

