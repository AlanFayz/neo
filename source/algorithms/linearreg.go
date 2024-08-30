package algorithms

// import "fmt"

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
	N float64
}

type RegressionModel struct {
	Coefficients []float64
	YIntercept float64

}

func SquareSum(data []float64) float64 {
	var sum float64
	for _, i := range data {
		sum += i * i
	}
	return sum

}

func Cut(i int, xs []float64) ([]float64) {
	fin := []float64{}
	for index,x := range xs {
		if index == i{
			continue
		}
		fin = append(fin,x)
	}
	return fin
//   return append(xs[:i], xs[i+1:]...)

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




func LinearRegression(f DataInput) RegressionModel{
	statData := StatisticalData{}

	for _, indepVar := range f.X {
		statData.SquareSums = append(statData.SquareSums, SquareSum(indepVar))
		statData.CrossYsums = append(statData.CrossYsums, CrossSums(indepVar, f.Y))
		statData.Sums = append(statData.Sums,SumArr(indepVar))
	}
	statData.DotProduct = DotProduct(f.X)
	statData.N = float64(len(f.Y))
	statData.YSum = SumArr(f.Y)
	regressionSums := RegressionSums{}
	// fmt.Println(statData)
	for i := range len(statData.Sums) {
		regressionSums.SumX2 =append(regressionSums.SumX2,statData.SquareSums[i]-((statData.Sums[i]*statData.Sums[i]))/statData.N)
		regressionSums.SumXy2 =append(regressionSums.SumXy2,statData.CrossYsums[i]-((statData.Sums[i]*statData.YSum)/statData.N))
	}
	regressionSums.SumXX = statData.DotProduct-Product(statData.Sums)/statData.N
	// fmt.Println(regressionSums)

	model := RegressionModel{}
	for i := range len(statData.Sums) {
		numerator := Product(Cut(i, regressionSums.SumX2))*regressionSums.SumXy2[i]-regressionSums.SumXX*Product(Cut(i,regressionSums.SumXy2))
		denominator := Product(regressionSums.SumX2)-(regressionSums.SumXX)*(regressionSums.SumXX)
		model.Coefficients = append(model.Coefficients, numerator/denominator)	
		} 
	
	var YIntercept float64
	YIntercept = statData.YSum/statData.N
	for index,coeff := range model.Coefficients {
		YIntercept-= (statData.Sums[index]/statData.N)*coeff
	}
	model.YIntercept = YIntercept
	return model
}


func (model *RegressionModel) Predict(inputs []float64) float64{
	var prediction float64
	prediction = model.YIntercept
	for index,coeff := range model.Coefficients {
		prediction += inputs[index]*coeff
	}
	
	return prediction
}

