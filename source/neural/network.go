package neural

import (
	"errors"

	internalmath "github.com/DevAlgos/neo/source/math"
)

type Layer struct {
	Neurons          []SigmoidNeuron
	DerivativeValues internalmath.Vector[float64]
	WeightedInputs   internalmath.Vector[float64]
	Activations      internalmath.Vector[float64]
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

	first.DerivativeValues = *internalmath.CreateVectorWithSize[float64](neuronCount[0])
	first.WeightedInputs = *internalmath.CreateVectorWithSize[float64](neuronCount[0])
	first.Activations = *internalmath.CreateVectorWithSize[float64](neuronCount[0])

	network.Layers = append(network.Layers, first)

	//do the rest of the layers hidden and output
	for i := 1; i < len(neuronCount); i++ {
		layer := Layer{}

		for neuronIndex := 0; neuronIndex < neuronCount[i]; neuronIndex++ {
			layer.Neurons = append(layer.Neurons, *CreateNeuronRandomized(neuronCount[i-1]))
		}

		layer.DerivativeValues = *internalmath.CreateVectorWithSize[float64](neuronCount[i])
		layer.WeightedInputs = *internalmath.CreateVectorWithSize[float64](neuronCount[i])
		layer.Activations = *internalmath.CreateVectorWithSize[float64](neuronCount[i])

		network.Layers = append(network.Layers, layer)
	}

	return &network
}

func (n *NeuralNetwork) GetOutputs() *internalmath.Vector[float64] {
	return &n.Layers[len(n.Layers)-1].Activations
}

func (n *NeuralNetwork) ComputeInput() {
	for i := 0; i < len(n.Layers[0].Neurons); i++ {
		n.Layers[0].WeightedInputs.Data[i] = n.Layers[0].Neurons[i].Compute(&n.Inputs)
		n.Layers[0].Activations.Data[i] = internalmath.Sigmoid(n.Layers[0].WeightedInputs.Data[i])
	}

}

func (n *NeuralNetwork) Train(data *DataGroup) {
	n.ComputeGradients(data.Input, data.Expected)
	n.AdjustWeights(data.LearningRate, 1.0)
}

func (n *NeuralNetwork) TrainNew(data *DataGroup, cycleCount float64) {
	n.ClearGradients()

	for i := 0; i < int(cycleCount); i++ {
		n.ComputeGradientsNew(data.Input, data.Expected)
		n.AdjustWeights(data.LearningRate, float64(i)+1.0)
	}

}

func (n *NeuralNetwork) CalculateCost(input *internalmath.Vector[float64], expectedOutput *internalmath.Vector[float64]) float64 {
	n.Compute(input)
	return internalmath.CostVector(expectedOutput, n.GetOutputs())
}

func (n *NeuralNetwork) ComputeGradients(input *internalmath.Vector[float64], expectedOutput *internalmath.Vector[float64]) {
	cost := n.CalculateCost(input, expectedOutput)

	const h float64 = 1e-10

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

func (n *NeuralNetwork) ComputeGradientsNew(input *internalmath.Vector[float64], expectedOutput *internalmath.Vector[float64]) {
	// output layer
	n.Compute(input)

	last := len(n.Layers) - 1
	outputLayer := &n.Layers[last]

	for i := 0; i < outputLayer.WeightedInputs.Size(); i++ {
		costActivationDerivative := internalmath.CostDerivative(expectedOutput.Data[i], outputLayer.Activations.Data[i])
		activationInputDerivative := internalmath.SigmoidDerivative(outputLayer.WeightedInputs.Data[i])

		outputLayer.DerivativeValues.Data[i] = costActivationDerivative * activationInputDerivative
	}

	// compute gradients for weights and biases on the output layer
	nextLayer := &n.Layers[last-1]
	for out := 0; out < len(outputLayer.Neurons); out++ {
		outNeuron := &outputLayer.Neurons[out]

		//differentiates to just one
		outNeuron.BiasGradient = 1 * outputLayer.DerivativeValues.Data[out]

		for prev := 0; prev < nextLayer.Activations.Size(); prev++ {
			// gradient for the weights connecting previous layer neuron to the output neuron
			weightGradient := outputLayer.DerivativeValues.Data[out] * nextLayer.Activations.Data[prev]

			outNeuron.Gradients.Data[prev] += weightGradient
		}

	}

	// backpropagation through hidden layers
	for layerIndex := len(n.Layers) - 2; layerIndex >= 0; layerIndex-- {
		currentLayer := &n.Layers[layerIndex]
		nextLayer := &n.Layers[layerIndex+1]

		// Compute the gradient of the activations for the current layer
		for i := 0; i < currentLayer.Activations.Size(); i++ {
			// delta for the current neuron
			delta := 0.0

			// sum over all the neurons in the next layer
			for j := 0; j < len(nextLayer.Neurons); j++ {
				//weight of the connection of the neuron on the previous layer to the current layers neuron/activation
				weight := nextLayer.Neurons[j].Weights.Data[i]
				delta += nextLayer.DerivativeValues.Data[j] * weight
			}

			activationInputDerivative := internalmath.SigmoidDerivative(currentLayer.Activations.Data[i])
			currentLayer.DerivativeValues.Data[i] = delta * activationInputDerivative
		}

		for i := 0; i < len(currentLayer.Neurons); i++ {
			neuron := &currentLayer.Neurons[i]

			neuron.BiasGradient = 0

			// Update gradients for each weight of the current neuron
			for j := 0; j < len(neuron.Gradients.Data); j++ {
				if j < len(nextLayer.Activations.Data) {
					// Gradient for the weights connecting current layer neuron to the next layer neuron
					weightGradient := currentLayer.DerivativeValues.Data[i] * nextLayer.Activations.Data[j]
					neuron.Gradients.Data[j] += weightGradient
				}
			}

			neuron.BiasGradient += currentLayer.DerivativeValues.Data[i]
		}
	}
}

func (n *NeuralNetwork) ClearGradients() {
	for i := 0; i < len(n.Layers); i++ {
		layer := &n.Layers[i]
		for j := 0; j < len(layer.Neurons); j++ {
			neuron := &layer.Neurons[j]

			for k := 0; k < neuron.Gradients.Size(); k++ {
				neuron.Gradients.Data[k] = 0
			}

			neuron.BiasGradient = 0
		}
	}
}

func (n *NeuralNetwork) AdjustWeights(learningRate float64, cycleCount float64) {
	for i := 0; i < len(n.Layers); i++ {
		layer := &n.Layers[i]

		for j := 0; j < len(layer.Neurons); j++ {
			neuron := &layer.Neurons[j]

			for k := 0; k < neuron.Weights.Size(); k++ {
				neuron.Weights.Data[k] -= (neuron.Gradients.Data[k] / cycleCount) * learningRate
			}

			neuron.Bias -= (neuron.BiasGradient / cycleCount) * learningRate
		}
	}
}

func (n *NeuralNetwork) ComputeLayers() {
	for i := 1; i < len(n.Layers); i++ {
		input := &n.Layers[i-1].Activations
		output := &n.Layers[i]

		for neuron := 0; neuron < len(output.Neurons); neuron++ {
			neuronValue := &output.Neurons[neuron]
			output.WeightedInputs.Data[neuron] = neuronValue.Compute(input)
			output.Activations.Data[neuron] = internalmath.Sigmoid(output.WeightedInputs.Data[neuron])
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
