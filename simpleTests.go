package main

import (
	"blueprint"
	"fmt"
	"math/rand"
	"time"
)

func simple1() {
	// Example JSON configuration for the neural network
	const neuronConfig = `
		[
			{"id": 1, "type": "input", "value": 0, "connections": []},
			{"id": 2, "type": "input", "value": 0, "connections": []},
			{"id": 3, "type": "dense", "bias": 0.1, "activation": "relu", "connections": [[1, 0.5], [2, 0.2]]},
			{"id": 4, "type": "dense", "bias": -0.1, "activation": "tanh", "connections": [[1, 0.3], [2, 0.8]]},
			{"id": 5, "type": "rnn", "bias": 0.0, "activation": "tanh", "connections": [[3, 1.0], [4, 1.0]]},
			{"id": 6, "type": "lstm", "bias": 0.0, "activation": "tanh", "connections": [[3, 1.0], [4, 1.0]]},
			{"id": 7, "type": "cnn", "bias": 1.0, "activation": "leaky_relu", "connections": [[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]]},
			{"id": 8, "type": "attention", "bias": 0.0, "activation": "linear", "connections": [[5, 1.0], [6, 1.0], [7, 1.0]]},
			{"id": 9, "type": "output", "bias": 0.0, "activation": "linear", "connections": [[8, 1.0]]}
		]
		`

	rand.Seed(time.Now().UnixNano())

	// Initialize neural network blueprint
	bp := blueprint.NewBlueprint()

	// Load neurons from JSON configuration
	err := bp.LoadNeurons(neuronConfig)
	if err != nil {
		fmt.Printf("Error loading neurons: %v\n", err)
		return
	}

	// Define input and output nodes
	bp.AddInputNodes([]int{1, 2})
	bp.AddOutputNodes([]int{9})

	// Define inputs
	inputs := map[int]float64{
		1: 1.5,
		2: -2.0,
	}

	// Run the network with specified timesteps for recurrent networks
	bp.RunNetwork(inputs, 3)
}
