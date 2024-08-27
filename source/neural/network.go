package neural

import (
	"errors"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type Layer struct {
	Neurons       []SigmoidNeuron
	ForwardInputs internalmath.Vector[float64]
}

type NeuralNetwork struct {
	Inputs       internalmath.Vector[float64]
	HiddenLayers []Layer
	Output       Layer
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

	network.HiddenLayers = append(network.HiddenLayers, first)

	//do the rest of the hidden layers
	for i := 1; i < len(neuronCount)-1; i++ {
		layer := Layer{}

		for neuronIndex := 0; neuronIndex < neuronCount[i]; neuronIndex++ {
			layer.Neurons = append(layer.Neurons, *CreateNeuronRandomized(neuronCount[i-1]))
		}

		network.HiddenLayers = append(network.HiddenLayers, layer)
	}

	//create the output layer
	for neuronIndex := 0; neuronIndex < neuronCount[len(neuronCount)-1]; neuronIndex++ {
		network.Output.Neurons = append(network.Output.Neurons, *CreateNeuronRandomized(neuronCount[len(neuronCount)-2]))
	}

	return &network
}

func (n *NeuralNetwork) Randomize() {
	for layerIndex := 0; layerIndex < len(n.HiddenLayers); layerIndex++ {
		layer := &n.HiddenLayers[layerIndex]
		for i := 0; i < len(layer.Neurons); i++ {
			layer.Neurons[i] = *CreateNeuronRandomized(layer.Neurons[i].Weights.Size())
		}
	}
}

func (n *NeuralNetwork) GetOutputs() *internalmath.Vector[float64] {
	return &n.Output.ForwardInputs
}

func (n *NeuralNetwork) ComputeInput() {
	for i := 0; i < len(n.HiddenLayers[0].Neurons); i++ {
		n.HiddenLayers[0].ForwardInputs.PushValue(n.HiddenLayers[0].Neurons[i].ComputeSigmoid(&n.Inputs))
	}
}

func (n *NeuralNetwork) ComputeHiddenLayers() {
	for i := 1; i < len(n.HiddenLayers); i++ {
		input := &n.HiddenLayers[i-1].ForwardInputs
		output := &n.HiddenLayers[i]

		for neuron := 0; neuron < len(output.Neurons); neuron++ {
			neuronValue := &output.Neurons[neuron]
			output.ForwardInputs.PushValue(neuronValue.ComputeSigmoid(input))
		}
	}
}

func (n *NeuralNetwork) ComputeOutputs() {
	last := len(n.HiddenLayers) - 1

	input := &n.HiddenLayers[last].ForwardInputs
	output := &n.Output

	for neuron := 0; neuron < len(output.Neurons); neuron++ {
		neuronValue := &output.Neurons[neuron]
		output.ForwardInputs.PushValue(neuronValue.ComputeSigmoid(input))
	}
}

func (n *NeuralNetwork) Compute() error {
	if len(n.HiddenLayers) == 0 || n.Inputs.Size() == 0 || len(n.Output.Neurons) == 0 {
		return errors.New("neural network not properly initialized")
	}

	n.ComputeInput()
	n.ComputeHiddenLayers()
	n.ComputeOutputs()

	return nil
}
