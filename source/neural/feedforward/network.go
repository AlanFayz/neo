package feedforward

type NeuralNetwork struct {
	LayerList   []Layer
	FinalOutput []float64
}

// should be arrays in the future
type DataGroup struct {
	Input        []float64
	Expected     []float64
	LearningRate float64
}

func CreateNeuralNetwork(inputCount int, neuronCount ...int) *NeuralNetwork {
	network := NeuralNetwork{}

	network.LayerList = append(network.LayerList, CreateLayer(inputCount, neuronCount[0]))

	for i := 1; i < len(neuronCount); i++ {
		network.LayerList = append(network.LayerList, CreateLayer(neuronCount[i-1], neuronCount[i]))
	}

	return &network
}

func (n *NeuralNetwork) FeedForward(input []float64) {
	forwarded := n.LayerList[0].FeedInput(input)

	for i := 1; i < len(n.LayerList); i++ {
		forwarded = n.LayerList[i].FeedForward(&n.LayerList[i-1])
	}

	n.FinalOutput = forwarded.Activations
}

func (n *NeuralNetwork) GetResult() []float64 {
	return n.FinalOutput
}

func (n *NeuralNetwork) ClearGradients() {
	for i := 0; i < len(n.LayerList); i++ {
		n.LayerList[i].ResetGradients()
	}
}

func (n *NeuralNetwork) Train(d *DataGroup) {
	n.FeedForward(d.Input)

	last := len(n.LayerList) - 1

	n.LayerList[last].ComputeDerivativesBackPropagationOutputLayer(d.Expected)
	n.LayerList[last].UpdateGradients()

	for i := last - 1; i >= 0; i-- {
		n.LayerList[i].ComputeDerivativesBackPropagation(&n.LayerList[i+1])
		n.LayerList[i].UpdateGradients()
	}

	for i := 0; i < len(n.LayerList); i++ {
		n.LayerList[i].ApplyGradients(d.LearningRate, 1)
	}
}
