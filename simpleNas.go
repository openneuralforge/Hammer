package main

import (
	"blueprint"
	"fmt"
	"math/rand"
	"time"
)

func simpleNAS() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Initialize neural network blueprint
	bp := blueprint.NewBlueprint()

	// Define input and output nodes
	bp.AddInputNodes([]int{1, 2})
	bp.AddOutputNodes([]int{3})

	// Add input neurons to the blueprint
	bp.Neurons[1] = &blueprint.Neuron{
		ID:   1,
		Type: "input",
	}
	bp.Neurons[2] = &blueprint.Neuron{
		ID:   2,
		Type: "input",
	}

	// Add output neuron to the blueprint
	bp.Neurons[3] = &blueprint.Neuron{
		ID:          3,
		Type:        "output",
		Activation:  "linear",
		Connections: [][]float64{}, // No initial connections
	}

	// Define sample data (sessions)
	sessions := []blueprint.Session{
		{
			InputVariables: map[int]float64{
				1: 0.5,
				2: 0.8,
			},
			ExpectedOutput: map[int]float64{
				3: 1.3,
			},
			Timesteps: 1,
		},
		{
			InputVariables: map[int]float64{
				1: -0.2,
				2: 0.4,
			},
			ExpectedOutput: map[int]float64{
				3: 0.2,
			},
			Timesteps: 1,
		},
		{
			InputVariables: map[int]float64{
				1: 0.0,
				2: 0.0,
			},
			ExpectedOutput: map[int]float64{
				3: 0.0,
			},
			Timesteps: 1,
		},
		// Add more sessions as needed
	}

	// Set parameters for SimpleNAS
	maxIterations := 20000
	forgivenessThreshold := 0.1 // 10%

	// Perform SimpleNAS
	bp.SimpleNAS(sessions, maxIterations, forgivenessThreshold)

	// Test the final model
	fmt.Println("Testing the final model:")
	for _, session := range sessions {
		bp.RunNetwork(session.InputVariables, session.Timesteps)
		predictedOutput := bp.GetOutputs()
		fmt.Printf("Input: %v, Expected Output: %v, Predicted Output: %v\n", session.InputVariables, session.ExpectedOutput, predictedOutput)
	}

	// Display the final model
	fmt.Println("Final model:")
	jsonStr, err := bp.ToJSON()
	if err != nil {
		fmt.Printf("Error converting blueprint to JSON: %v\n", err)
	} else {
		fmt.Println(jsonStr)
	}
}

func simpleNASWithoutCrossover() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Initialize neural network blueprint
	bp := blueprint.NewBlueprint()

	// Define input and output nodes
	bp.AddInputNodes([]int{1, 2})
	bp.AddOutputNodes([]int{3})

	// Add input neurons to the blueprint
	bp.Neurons[1] = &blueprint.Neuron{
		ID:   1,
		Type: "input",
	}
	bp.Neurons[2] = &blueprint.Neuron{
		ID:   2,
		Type: "input",
	}

	// Add output neuron to the blueprint
	bp.Neurons[3] = &blueprint.Neuron{
		ID:          3,
		Type:        "output",
		Activation:  "linear",
		Connections: [][]float64{}, // No initial connections
	}

	// Define sample data (sessions)
	sessions := []blueprint.Session{
		{
			InputVariables: map[int]float64{
				1: 0.5,
				2: 0.8,
			},
			ExpectedOutput: map[int]float64{
				3: 1.3,
			},
			Timesteps: 1,
		},
		{
			InputVariables: map[int]float64{
				1: -0.2,
				2: 0.4,
			},
			ExpectedOutput: map[int]float64{
				3: 0.2,
			},
			Timesteps: 1,
		},
		{
			InputVariables: map[int]float64{
				1: 0.0,
				2: 0.0,
			},
			ExpectedOutput: map[int]float64{
				3: 0.0,
			},
			Timesteps: 1,
		},
		// Add more sessions as needed
	}

	// Set parameters for SimpleNASWithoutCrossover
	maxIterations := 200000
	forgivenessThreshold := 0.1 // 10%

	// Randomly select a neuron type to add
	neuronTypes := []string{"dense", "attention", "nca"}

	metrics := []string{"exact"}

	// Perform SimpleNASWithoutCrossover
	bp.SimpleNASWithoutCrossover(sessions, maxIterations, forgivenessThreshold, neuronTypes, metrics)

	// Test the final model
	fmt.Println("Testing the final model:")
	for _, session := range sessions {
		bp.RunNetwork(session.InputVariables, session.Timesteps)
		predictedOutput := bp.GetOutputs()
		fmt.Printf("Input: %v, Expected Output: %v, Predicted Output: %v\n", session.InputVariables, session.ExpectedOutput, predictedOutput)
	}

	// Display the final model
	fmt.Println("Final model:")
	jsonStr, err := bp.ToJSON()
	if err != nil {
		fmt.Printf("Error converting blueprint to JSON: %v\n", err)
	} else {
		fmt.Println(jsonStr)
	}
}
