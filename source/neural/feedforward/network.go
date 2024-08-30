package feedforward

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
)

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

type Batch struct {
	DataList []DataGroup
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

	for i := last - 1; i >= 0; i-- {
		n.LayerList[i].ComputeDerivativesBackPropagation(&n.LayerList[i+1])
	}

	for i := 0; i < len(n.LayerList); i++ {
		n.LayerList[i].UpdateGradients()
		n.LayerList[i].ApplyGradients(d.LearningRate, 1)
	}
}

func (n *NeuralNetwork) TrainWithoutApplyingGradients(d *DataGroup) {
	n.FeedForward(d.Input)

	last := len(n.LayerList) - 1

	n.LayerList[last].ComputeDerivativesBackPropagationOutputLayer(d.Expected)

	for i := last - 1; i >= 0; i-- {
		n.LayerList[i].ComputeDerivativesBackPropagation(&n.LayerList[i+1])
	}

	for i := 0; i < len(n.LayerList); i++ {
		n.LayerList[i].UpdateGradients()
	}

}

func (n *NeuralNetwork) Combine(other *NeuralNetwork) {

	for i := 0; i < len(other.LayerList); i++ {
		n.LayerList[i].Combine(&other.LayerList[i])
	}

}

// learning rate in dataGroups not used
func (n *NeuralNetwork) TrainDataParallel(dataGroups []DataGroup, learningRate float64, cycleCount int) {

	//need to distribute the work across all the cores
	//for each pass
	coreCount := runtime.NumCPU()

	distribution := max(len(dataGroups)/coreCount, 1)
	fmt.Println(distribution)

	for cycle := 0; cycle < cycleCount; cycle++ {
		//create the copies
		copies := make([]NeuralNetwork, 0)

		for i := 0; i < coreCount; i++ {
			copies = append(copies, *n)
		}

		var waitGroup sync.WaitGroup

		//first randomize the dataGroups

		rand.Shuffle(len(dataGroups), func(i, j int) {
			dataGroups[i], dataGroups[j] = dataGroups[j], dataGroups[i]
		})

		//run each network
		for core := 0; core < coreCount; core++ {
			start := distribution * core
			end := min(start+distribution, len(dataGroups))

			waitGroup.Add(1)
			go func(index, start, end int) {
				defer waitGroup.Done()

				currentNetwork := &copies[index]

				for i := start; i < end; i++ {
					currentNetwork.TrainWithoutApplyingGradients(&dataGroups[i])
				}

			}(core, start, end)
		}

		waitGroup.Wait()

		//now we need to combine the networks gradients.
		for i := 0; i < coreCount; i++ {
			n.Combine(&copies[i])
		}

		//apply gradients
		for i := 0; i < len(n.LayerList); i++ {
			n.LayerList[i].ApplyGradients(learningRate, 1)
		}
	}
}
