package main

import (
	"blueprint"
	"fmt"
)

func createAndSaveAllNeuronTypesToFile(destination string) {
	// Define the neuron types to be included
	neuronTypes := []string{
		"dense",
		"rnn",
		"lstm",
		"cnn",
		"dropout",
		"batch_norm",
		"attention",
		"nca",
	}

	// Initialize a new blueprint
	bp := blueprint.NewBlueprint()

	// Add dummy input and output neurons for testing
	bp.AddInputNodes([]int{1, 2})
	bp.AddOutputNodes([]int{3})

	// Insert one neuron of each type
	for _, neuronType := range neuronTypes {
		err := bp.InsertNeuronOfTypeBetweenInputsAndOutputs(neuronType)
		if err != nil {
			fmt.Printf("Error inserting neuron of type '%s': %v\n", neuronType, err)
			return
		}
	}

	// Save the network as JSON to the specified destination
	err := bp.SaveToJSON(destination)
	if err != nil {
		fmt.Printf("Error saving Blueprint to file: %v\n", err)
		return
	}

	fmt.Printf("Neural network with all neuron types saved successfully to '%s'\n", destination)
}
