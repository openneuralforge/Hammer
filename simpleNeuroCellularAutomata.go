package main

import (
	"blueprint"
	"fmt"
	"math/rand"
	"time"
)

func testNeuroCellularAutomata() {
	// Example JSON configuration for a network with NCA neurons
	const neuronConfig = `
		[
			{"id": 1, "type": "input", "value": 0, "connections": []},
			{"id": 2, "type": "input", "value": 0, "connections": []},
			{"id": 3, "type": "nca", "bias": 0.1, "activation": "relu", "neighborhood": [1, 2], "update_rules": "average"},
			{"id": 4, "type": "nca", "bias": -0.1, "activation": "tanh", "neighborhood": [1, 2, 3], "update_rules": "sum"},
			{"id": 5, "type": "dense", "bias": 0.0, "activation": "linear", "connections": [[3, 0.5], [4, 0.8]]},
			{"id": 6, "type": "output", "bias": 0.0, "activation": "linear", "connections": [[5, 1.0]]}
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
	bp.AddOutputNodes([]int{6})

	// Define inputs
	inputs := map[int]float64{
		1: 2.0,
		2: -1.0,
	}

	// Run the network with specified timesteps
	fmt.Println("Running Neuro Cellular Automata Test...")
	bp.RunNetwork(inputs, 5) // Adjust timesteps for NCA to observe temporal evolution
}

func testFullRangeOfNeuronsNCA() {
	// Updated JSON configuration for a network with all neuron types
	const neuronConfig = `
		[
			{"id": 1, "type": "input", "value": 0, "connections": []},
			{"id": 2, "type": "input", "value": 0, "connections": []},
			{"id": 3, "type": "dense", "bias": 0.1, "activation": "relu", "connections": [[1, 0.5], [2, 0.3]]},
			{"id": 4, "type": "rnn", "bias": -0.2, "activation": "tanh", "connections": [[3, 0.6]]},
			{"id": 5, "type": "lstm", "bias": 0.0, "activation": "sigmoid", "connections": [[3, 0.4], [4, 0.5]]},
			{"id": 6, "type": "cnn", "bias": 0.3, "activation": "relu", "connections": [[1, 1.0], [2, 1.0]]},
			{"id": 7, "type": "nca", "bias": 0.0, "activation": "relu", "neighborhood": [3, 4], "update_rules": "sum"},
			{"id": 8, "type": "attention", "bias": 0.0, "activation": "linear", "connections": [[5, 1.0], [6, 1.0], [7, 1.0]]},
			{"id": 9, "type": "dropout", "bias": 0.0, "activation": "linear", "dropout_rate": 0.3, "connections": [[8, 1.0]]},
			{"id": 10, "type": "batch_norm", "bias": 0.0, "activation": "linear", "connections": [[9, 1.0]]},
			{"id": 11, "type": "output", "bias": 0.0, "activation": "linear", "connections": [[10, 1.0]]}
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
	bp.AddOutputNodes([]int{11})

	// Define inputs
	inputs := map[int]float64{
		1: 1.5,
		2: -2.0,
	}

	// Run the network with specified timesteps
	fmt.Println("---Testing Full Range of Neurons---")
	bp.RunNetwork(inputs, 5)
}

func testNeuroCellularAutomataWithCNNKernels() {
	fmt.Println("---NeuroCellularAutomataWithCNNKernels---")

	// Example JSON configuration for a network with CNN neurons having multiple kernels and NCA neurons
	const neuronConfig = `
    [
        {"id": 1, "type": "input"},
        {"id": 2, "type": "input"},
        {
            "id": 3,
            "type": "cnn",
            "bias": 0.1,
            "activation": "relu",
            "connections": [[1, 0.5], [2, 0.5]],
            "kernels": [
                [0.2, 0.5],
                [0.3, 0.4]
            ]
        },
        {
            "id": 4,
            "type": "nca",
            "bias": 0.0,
            "activation": "relu",
            "neighborhood": [3],
            "update_rules": "sum"
        },
        {
            "id": 5,
            "type": "output",
            "bias": 0.0,
            "activation": "linear",
            "connections": [[4, 1.0]]
        }
    ]
    `

	rand.Seed(time.Now().UnixNano())

	// Initialize neural network blueprint
	bp := blueprint.NewBlueprint()
	bp.ScalarActivationMap = blueprint.InitializeActivationFunctions()

	// Load neurons from JSON configuration
	err := bp.LoadNeurons(neuronConfig)
	if err != nil {
		fmt.Printf("Error loading neurons: %v\n", err)
		return
	}

	// Define input and output nodes
	bp.AddInputNodes([]int{1, 2})
	bp.AddOutputNodes([]int{5})

	// Define inputs that ensure at least one kernel produces a positive sum
	inputs := map[int]float64{
		1: 2.0,
		2: 1.0,
	}

	// Run the network with specified timesteps
	fmt.Println("Running Neuro Cellular Automata with CNN Kernels Test...")
	bp.RunNetwork(inputs, 3) // Adjust timesteps as needed
}
