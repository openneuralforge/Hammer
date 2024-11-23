package main

import "fmt"

func main() {
	fmt.Println("---SIMPLE---")
	simple1()
	fmt.Println("---Quantum---")
	RunQuantumExample()
	fmt.Println("---NeuroCellular Automata---")
	testNeuroCellularAutomata()
	fmt.Println("---testFullRangeOfNeuronsNCA---")
	testFullRangeOfNeuronsNCA()
	fmt.Println("---testNeuroCellularAutomataWithCNNKernels---")
	testNeuroCellularAutomataWithCNNKernels()

	fmt.Println("---testMutations---")
	testMutations()
	fmt.Println("---testMutationsWithMultipleTypes---")
	testMutationsWithMultipleTypes()
	fmt.Println("---testAllNeuronTypes---")
	testAllNeuronTypes()
}
