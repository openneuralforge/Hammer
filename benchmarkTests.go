package main

import (
	"blueprint"
	"fmt"
	"time"
)

// TestRunBenchmark runs a benchmark for the AI framework and prints the results.
func TestRunBenchmark() {
	// Initialize a Blueprint instance
	bp := blueprint.NewBlueprint()

	// Set the benchmark duration
	benchmarkDuration := 10 * time.Second // Adjust as needed

	fmt.Println("Starting benchmark for the AI framework...")
	formattedOps32Single, formattedOps64Single, formattedOps32Multi, formattedOps64Multi, maxLayers32Single, maxLayers64Single, maxLayers32Multi, maxLayers64Multi := bp.RunBenchmark(benchmarkDuration)

	// Print the benchmark results
	fmt.Println("\nBenchmark Results:")
	fmt.Printf("Float32 Single-threaded Ops/sec: %s\n", formattedOps32Single)
	fmt.Printf("Float64 Single-threaded Ops/sec: %s\n", formattedOps64Single)
	fmt.Printf("Float32 Multi-threaded Ops/sec: %s\n", formattedOps32Multi)
	fmt.Printf("Float64 Multi-threaded Ops/sec: %s\n", formattedOps64Multi)
	fmt.Printf("Max Float32 Single-threaded Layers: %s\n", maxLayers32Single)
	fmt.Printf("Max Float64 Single-threaded Layers: %s\n", maxLayers64Single)
	fmt.Printf("Max Float32 Multi-threaded Layers: %s\n", maxLayers32Multi)
	fmt.Printf("Max Float64 Multi-threaded Layers: %s\n", maxLayers64Multi)
	fmt.Println("\nBenchmark complete.")
}
