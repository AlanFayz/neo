package algorithms

import "fmt"

// "gonum.org/v1/plot"
// "gonum.org/v1/plot/plotter"
// "gonum.org/v1/plot/plotutil"
// "gonum.org/v1/plot/vg"
type DataInput struct {
	Y []float64
	X [][]float64
}

type RegressionSums struct {
	SumX2 []float64
	SumXy2 []float64
	SumXX float64
}

type StatisticalData struct {
	DotProduct float64
	CrossYsums []float64
	SquareSums []float64
	Sums []float64
	YSum float64 
	N int
}

func SquareSum(data []float64) float64 {
	var sum float64
	for _, i := range data {
		sum += i * i
	}
	return sum

}

func CrossSums(x []float64, y []float64) float64 {
	var sum float64
	for index, i := range x {
		sum += i * y[index]
	}
	return sum

}
func SumArr(x []float64) float64 {
	var sum float64
	for _,i := range x {
		sum+=i
	}
	return sum
}


func Product(input []float64) float64 {
	var total float64
	total = 1
	for _,i := range input {
		total*= i
	}
	return total
}
func DotProduct(input [][]float64) float64 {
	var sum float64
	var curr float64

	for index := range input[0] {
		curr = 1
		for _, arr := range input {
			curr *= arr[index]
		}
		sum += curr

	}
	return sum
}




func LinearRegression(f DataInput){
	statData := StatisticalData{}

	for _, indepVar := range f.X {
		statData.SquareSums = append(statData.SquareSums, SquareSum(indepVar))
		statData.CrossYsums = append(statData.CrossYsums, CrossSums(indepVar, f.Y))
		statData.Sums = append(statData.Sums,SumArr(indepVar))
	}
	statData.DotProduct = DotProduct(f.X)
	statData.N = len(f.Y)
	statData.YSum = SumArr(f.Y)
	regressionSums := RegressionSums{}
	fmt.Println(statData)
	for i := range len(statData.Sums) {
		regressionSums.SumX2 =append(regressionSums.SumX2,statData.SquareSums[i]-((statData.Sums[i]*statData.Sums[i]))/float64(statData.N))
		regressionSums.SumXy2 =append(regressionSums.SumXy2,statData.CrossYsums[i]-((statData.Sums[i]*statData.YSum)/float64(statData.N)))
	}
	regressionSums.SumXX = statData.DotProduct-Product(statData.Sums)/float64(statData.N)
	fmt.Println(regressionSums)
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
