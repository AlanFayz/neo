package neural

import (
	"errors"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type Layer struct {
	Neurons         []SigmoidNeuron
	ForwardedInputs internalmath.Vector[float64]
}

type NeuralNetwork struct {
	Inputs internalmath.Vector[float64]
	Layers []Layer
}

// should be arrays in the future
type DataGroup struct {
	Input        *internalmath.Vector[float64]
	Expected     *internalmath.Vector[float64]
	LearningRate float64
}

// neuron count is the number of neurons per hidden layer, the last neuron count is for the output layer
func CreateNeuralNetwork(input *internalmath.Vector[float64], neuronCount ...int) *NeuralNetwork {
	network := NeuralNetwork{}

	//first we copy the inputs and make the first hidden layer
	network.Inputs = *input.Copy()

	first := Layer{}

	for neuronIndex := 0; neuronIndex < neuronCount[0]; neuronIndex++ {
		first.Neurons = append(first.Neurons, *CreateNeuronRandomized(network.Inputs.Size()))
	}

	first.ForwardedInputs = *internalmath.CreateVectorWithSize[float64](neuronCount[0])

	network.Layers = append(network.Layers, first)

	//do the rest of the layers hidden and output
	for i := 1; i < len(neuronCount); i++ {
		layer := Layer{}

		for neuronIndex := 0; neuronIndex < neuronCount[i]; neuronIndex++ {
			layer.Neurons = append(layer.Neurons, *CreateNeuronRandomized(neuronCount[i-1]))
		}

		layer.ForwardedInputs = *internalmath.CreateVectorWithSize[float64](neuronCount[i])
		network.Layers = append(network.Layers, layer)
	}

	return &network
}

func (n *NeuralNetwork) GetOutputs() *internalmath.Vector[float64] {
	return &n.Layers[len(n.Layers)-1].ForwardedInputs
}

func (n *NeuralNetwork) ComputeInput() {
	for i := 0; i < len(n.Layers[0].Neurons); i++ {
		n.Layers[0].ForwardedInputs.Data[i] = n.Layers[0].Neurons[i].ComputeSigmoid(&n.Inputs)
	}

}

func (n *NeuralNetwork) Train(data *DataGroup) {
	n.ComputeGradients(data.Input, data.Expected)
	n.AdjustWeights(data.LearningRate)
}

func (n *NeuralNetwork) CalculateCost(input *internalmath.Vector[float64], expectedOutput *internalmath.Vector[float64]) float64 {
	n.Compute(input)
	return internalmath.Cost(expectedOutput, n.GetOutputs())
}

func (n *NeuralNetwork) ComputeGradients(input *internalmath.Vector[float64], expectedOutput *internalmath.Vector[float64]) {
	cost := n.CalculateCost(input, expectedOutput)

	const h float64 = 1e-3

	for i := 0; i < len(n.Layers); i++ {
		layer := &n.Layers[i]

		for j := 0; j < len(layer.Neurons); j++ {
			neuron := &layer.Neurons[j]

			for k := 0; k < neuron.Weights.Size(); k++ {
				neuron.Weights.Data[k] += h
				newCost := n.CalculateCost(input, expectedOutput)
				neuron.Gradients.Data[k] = (newCost - cost) / h
				neuron.Weights.Data[k] -= h
			}

			neuron.Bias += h
			newCost := n.CalculateCost(input, expectedOutput)
			neuron.BiasGradient = (newCost - cost) / h
			neuron.Bias -= h
		}
	}
}

func (n *NeuralNetwork) AdjustWeights(learningRate float64) {
	for i := 0; i < len(n.Layers); i++ {
		layer := &n.Layers[i]

		for j := 0; j < len(layer.Neurons); j++ {
			neuron := &layer.Neurons[j]

			for k := 0; k < neuron.Weights.Size(); k++ {
				neuron.Weights.Data[k] -= neuron.Gradients.Data[k] * learningRate
			}

			neuron.Bias -= neuron.BiasGradient * learningRate
		}
	}
}

func (n *NeuralNetwork) ComputeLayers() {
	for i := 1; i < len(n.Layers); i++ {
		input := &n.Layers[i-1].ForwardedInputs
		output := &n.Layers[i]

		for neuron := 0; neuron < len(output.Neurons); neuron++ {
			neuronValue := &output.Neurons[neuron]
			output.ForwardedInputs.Data[neuron] = neuronValue.ComputeSigmoid(input)
		}

	}
}

func (n *NeuralNetwork) Compute(input *internalmath.Vector[float64]) error {
	if len(n.Layers) == 0 || n.Inputs.Size() == 0 {
		return errors.New("neural network not properly initialized")
	}

	n.Inputs = *input
	n.ComputeInput()
	n.ComputeLayers()

	return nil
}
