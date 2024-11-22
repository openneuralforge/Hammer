package main

import (
	"blueprint"
	"fmt"
)

func testMutations() {
	// Example JSON configuration for a simple network
	const neuronConfig = `
		[
			{"id": 1, "type": "input", "value": 0, "connections": []},
			{"id": 2, "type": "input", "value": 0, "connections": []},
			{"id": 3, "type": "dense", "bias": 0.1, "activation": "relu", "connections": [[1, 0.5], [2, 0.3]]},
			{"id": 4, "type": "output", "bias": 0.0, "activation": "linear", "connections": [[3, 1.0]]}
		]
	`

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
	bp.AddOutputNodes([]int{4})

	// Insert a new neuron of type "lstm" between inputs and outputs
	err = bp.InsertNeuronOfTypeBetweenInputsAndOutputs("lstm")
	if err != nil {
		fmt.Printf("Error inserting neuron: %v\n", err)
		return
	}

	// Define inputs
	inputs := map[int]float64{
		1: 1.5,
		2: -2.0,
	}

	// Run the network with specified timesteps
	fmt.Println("---Running Network After Inserting LSTM Neuron---")
	bp.RunNetwork(inputs, 3)
}

func testMutationsWithMultipleTypes() {
	// Example JSON configuration for a simple network
	const neuronConfig = `
		[
			{"id": 1, "type": "input", "value": 0, "connections": []},
			{"id": 2, "type": "input", "value": 0, "connections": []},
			{"id": 3, "type": "dense", "bias": 0.1, "activation": "relu", "connections": [[1, 0.5], [2, 0.3]]},
			{"id": 4, "type": "output", "bias": 0.0, "activation": "linear", "connections": [[3, 1.0]]}
		]
	`

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
	bp.AddOutputNodes([]int{4})

	// Perform multiple mutations: insert one neuron of each supported type
	err = bp.MutateNetwork()
	if err != nil {
		fmt.Printf("Error mutating network: %v\n", err)
		return
	}

	// Define inputs
	inputs := map[int]float64{
		1: 1.0,
		2: -1.5,
	}

	// Run the network with specified timesteps
	fmt.Println("---Running Network After Multiple Mutations---")
	bp.RunNetwork(inputs, 5)
}
