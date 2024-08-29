package main

import (
	// "gonum.org/v1/plot"
	// "gonum.org/v1/plot/plotter"
	// "gonum.org/v1/plot/plotutil"
	// "gonum.org/v1/plot/vg"
	"fmt"
)
type DataInput struct {
	Y []float64;
	X [][]float64;
}

type RegressionSums struct{
	ex2 []float64;
	eX2 []float64;
	EX2 []float64;

}

type StatisticalData struct {
	DotProduct float64;
	crossYsums []float64;
	squareSums []float64;
	n int;
}


func squareSum(data []float64) float64 { 
	var sum float64
	for _,i := range data {
		sum+= i*i
	}
	return sum

}

func crossSums(x []float64, y []float64) float64 {
	var sum float64
	for index,i := range x {
		sum+= i*y[index]
	}
	return sum
	
}

func DotProduct(input [][]float64) float64 {
	var sum float64
	var curr float64

	for index := range input[0]{
		curr = 1
		for _,arr := range input{
			curr*= arr[index]
		}
		sum+=curr

	}
	return sum
}
func main(){
	fmt.Println("Hi")
	y := []float64{140,155,159,179,192,200,212,215}
	x1 := []float64{60,62,67,70,71,72,75,78}
	x2 := []float64{22,25,24,20,15,14,14,11}
	comb  := [][]float64{x1, x2}

	f := DataInput{Y:y,X:comb}
	statData := StatisticalData{}

	for _, indepVar := range f.X{
		statData.squareSums = append(statData.squareSums,squareSum(indepVar))
		statData.crossYsums = append(statData.crossYsums,crossSums(indepVar,f.Y))
	}
	statData.DotProduct = DotProduct(f.X)
	statData.n = len(f.Y)
	fmt.Println(statData)
	// var RegSums RegressionSums
	

	// fin := []DataPoint{}
	// for index, y := range y{
	// 	t:=[]float64{float64(x1[index]), float64(x2[index])}
	// 	fin = append(fin, DataPoint{Y:float64(y),X:t})
	// }


}


// func main() {

// 	p := plot.New()
// 	points := []DataPoint{DataPoint{1,1},DataPoint{2,2},DataPoint{3,3},DataPoint{5,10}}
// 	fmt.Println(points)
// 	p.Title.Text = "Plotutil example"
// 	p.X.Label.Text = "X"
// 	p.Y.Label.Text = "Y"

// 	// err := plotutil.AddLinePoints(p,
// 	// 	"First", Points(points),

// 	// )
// 	scatter, err := plotter.NewScatter( Points(points))
// 	// scatter.GlyphStyle.Shape = plotter.CircleGlyph{}

// 	p.Add(scatter)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// Save the plot to a PNG file.
// 	if err := p.Save(10*vg.Inch, 10*vg.Inch, "points.png"); err != nil {
// 		panic(err)
// 	}
// }


// // randomPoints returns some random x, y points.
// func Points(points []DataPoint) plotter.XYs {
// 	pts := make(plotter.XYs, len(points))
// 	for index,p := range pts {
// 		// p := points[index]
// 		pts[index].X = p.X
// 		pts[index].Y = p.Y

// 	}
// 	return pts
// }