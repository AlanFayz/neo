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

func CreateNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{}
}

func (n *NeuralNetwork) SetInputs(inputs *internalmath.Vector[float64]) {
	n.Inputs = *inputs
}

func (n *NeuralNetwork) PushHiddenLayer(layer *Layer) {
	n.HiddenLayers = append(n.HiddenLayers, *layer)
}

func (n *NeuralNetwork) SetOutputCount(count int) {
	n.Output = Layer{Neurons: make([]SigmoidNeuron, count)}
}

func (n *NeuralNetwork) SetOutputNeuron(s *SigmoidNeuron, index int) {
	n.Output.Neurons[index] = *s
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
		return errors.New("Neural network not properly initialized")
	}

	n.ComputeInput()
	n.ComputeHiddenLayers()
	n.ComputeOutputs()

	return nil
}
