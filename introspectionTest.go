package main

import (
	"blueprint"
	"fmt"
)

func testBlueprintMethods() {
	// Create an instance of the Blueprint struct
	bp := blueprint.NewBlueprint()

	// Retrieve methods metadata as JSON
	methodsJSON, err := bp.GetBlueprintMethodsJSON()
	if err != nil {
		fmt.Printf("Error retrieving Blueprint methods: %v\n", err)
		return
	}

	// Display the JSON output
	fmt.Println("Blueprint Methods (JSON):")
	fmt.Println(methodsJSON)

	// Alternatively, retrieve the raw MethodInfo structs
	methods, err := bp.GetBlueprintMethods()
	if err != nil {
		fmt.Printf("Error retrieving Blueprint methods: %v\n", err)
		return
	}

	// Print method details in a human-readable format
	fmt.Println("\nBlueprint Methods (Readable):")
	for _, method := range methods {
		fmt.Printf("Method: %s\n", method.MethodName)
		fmt.Println("Parameters:")
		for _, param := range method.Parameters {
			fmt.Printf("  - Name: %s, Type: %s\n", param.Name, param.Type)
		}
		fmt.Println()
	}
}
