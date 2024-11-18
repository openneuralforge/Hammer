package main

import (
	"fmt"
	"math/rand"
	"time"

	"blueprint" // Import the blueprint package
)

func RunQuantumExample() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Initialize the neural network blueprint
	bp := blueprint.NewBlueprint()

	// Create quantum neurons
	quantumNeuron1 := &blueprint.QuantumNeuron{
		ID: 100, // Assign a unique ID
		QuantumState: blueprint.QuantumState{
			Amplitude: complex(1, 0), // Initial amplitude
			Phase:     0.0,           // Initial phase
		},
		QuantumGates: []blueprint.QuantumGate{
			{Type: "Hadamard"}, // Apply Hadamard gate
		},
		Entanglements: []blueprint.EntanglementInfo{},
		Superposition: []complex128{},
		Connections:   [][]complex128{},
	}

	quantumNeuron2 := &blueprint.QuantumNeuron{
		ID: 101,
		QuantumState: blueprint.QuantumState{
			Amplitude: complex(1, 0),
			Phase:     0.0,
		},
		QuantumGates: []blueprint.QuantumGate{
			{Type: "PauliX"}, // Apply Pauli-X gate
		},
		Entanglements: []blueprint.EntanglementInfo{},
		Superposition: []complex128{},
		Connections:   [][]complex128{},
	}

	// Add quantum neurons to the blueprint
	bp.QuantumNeurons[quantumNeuron1.ID] = quantumNeuron1
	bp.QuantumNeurons[quantumNeuron2.ID] = quantumNeuron2

	// Optionally, define entanglement between quantum neurons
	quantumNeuron1.Entanglements = []blueprint.EntanglementInfo{
		{
			PartnerID: quantumNeuron2.ID,
			Type:      "Bell",
			Strength:  1.0,
		},
	}

	// Process quantum neurons
	bp.ProcessQuantumNeuron(quantumNeuron1)
	bp.ProcessQuantumNeuron(quantumNeuron2)

	// Output the results
	fmt.Printf("Quantum Neuron %d final state: Amplitude=%v, Phase=%f\n", quantumNeuron1.ID, quantumNeuron1.QuantumState.Amplitude, quantumNeuron1.QuantumState.Phase)
	fmt.Printf("Quantum Neuron %d final state: Amplitude=%v, Phase=%f\n", quantumNeuron2.ID, quantumNeuron2.QuantumState.Amplitude, quantumNeuron2.QuantumState.Phase)
}

func RunQuantumExampleWithIntegration() {

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

	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Initialize the neural network blueprint
	bp := blueprint.NewBlueprint()

	// Load existing neurons from JSON configuration (if any)
	err := bp.LoadNeurons(neuronConfig)
	if err != nil {
		fmt.Printf("Error loading neurons: %v\n", err)
		return
	}

	// Define input and output nodes for the classical network
	bp.AddInputNodes([]int{1, 2}) // Input neurons (IDs 1 and 2)
	bp.AddOutputNodes([]int{9})   // Output neuron (ID 9)

	// Create quantum neurons
	quantumNeuron1 := &blueprint.QuantumNeuron{
		ID: 100, // Assign a unique ID not conflicting with classical neurons
		QuantumState: blueprint.QuantumState{
			Amplitude: complex(1, 0), // Initial amplitude
			Phase:     0.0,           // Initial phase
		},
		QuantumGates: []blueprint.QuantumGate{
			{Type: "Hadamard"}, // Apply Hadamard gate
		},
		Entanglements: []blueprint.EntanglementInfo{},
		Superposition: []complex128{},
		Connections:   [][]complex128{},
	}

	quantumNeuron2 := &blueprint.QuantumNeuron{
		ID: 101,
		QuantumState: blueprint.QuantumState{
			Amplitude: complex(1, 0),
			Phase:     0.0,
		},
		QuantumGates: []blueprint.QuantumGate{
			{Type: "PauliX"}, // Apply Pauli-X gate
		},
		Entanglements: []blueprint.EntanglementInfo{},
		Superposition: []complex128{},
		Connections:   [][]complex128{},
	}

	// Add quantum neurons to the blueprint's QuantumNeurons map
	bp.QuantumNeurons[quantumNeuron1.ID] = quantumNeuron1
	bp.QuantumNeurons[quantumNeuron2.ID] = quantumNeuron2

	// Optionally, define entanglement between quantum neurons
	quantumNeuron1.Entanglements = []blueprint.EntanglementInfo{
		{
			PartnerID: quantumNeuron2.ID,
			Type:      "Bell",
			Strength:  1.0,
		},
	}

	// Process quantum neurons
	bp.ProcessQuantumNeuron(quantumNeuron1)
	bp.ProcessQuantumNeuron(quantumNeuron2)

	// Obtain measured values from quantum neurons
	measuredValue1 := real(quantumNeuron1.QuantumState.Amplitude)
	measuredValue2 := real(quantumNeuron2.QuantumState.Amplitude)

	fmt.Printf("Quantum Neuron %d measured value: %f\n", quantumNeuron1.ID, measuredValue1)
	fmt.Printf("Quantum Neuron %d measured value: %f\n", quantumNeuron2.ID, measuredValue2)

	// Integrate measured values into the classical network as inputs
	// Update the values of the existing input neurons (IDs 1 and 2)
	if neuron, exists := bp.Neurons[1]; exists && neuron.Type == "input" {
		neuron.Value = measuredValue1
	} else {
		fmt.Printf("Input Neuron %d not found or not an input neuron.\n", 1)
		return
	}

	if neuron, exists := bp.Neurons[2]; exists && neuron.Type == "input" {
		neuron.Value = measuredValue2
	} else {
		fmt.Printf("Input Neuron %d not found or not an input neuron.\n", 2)
		return
	}

	// Run the classical network for a specified number of timesteps
	timesteps := 3
	bp.Forward(nil, timesteps)

	// Retrieve outputs from the network
	outputs := bp.GetOutputs()
	fmt.Println("Final Outputs from Classical Network:")
	for id, value := range outputs {
		fmt.Printf("Neuron %d: %f\n", id, value)
	}
}
