package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"blueprint"
)

const (
	baseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir  = "mnist"
	modelDir  = "models" // New directory for saving models
	modelName = "mnist_model.json"
)

func simpleMnist() {
	bp := blueprint.NewBlueprint()

	// Ensure MNIST data is downloaded and unzipped
	if err := EnsureMNISTDownloads(bp, mnistDir); err != nil {
		log.Fatalf("Failed to ensure MNIST downloads: %v", err)
	}

	// Process the MNIST data
	imageFile := filepath.Join(mnistDir, "train-images-idx3-ubyte")
	labelFile := filepath.Join(mnistDir, "train-labels-idx1-ubyte")
	outputDir := filepath.Join(mnistDir, "output")

	if err := UnpackMNIST(imageFile, labelFile, outputDir); err != nil {
		log.Fatalf("Failed to unpack MNIST data: %v", err)
	}

	// Train the model
	if err := TrainOnMNIST(bp, outputDir); err != nil {
		log.Fatalf("Failed to train on MNIST data: %v", err)
	}
}

// EnsureMNISTDownloads downloads and unzips the MNIST dataset.
func EnsureMNISTDownloads(bp *blueprint.Blueprint, targetDir string) error {
	if err := os.MkdirAll(targetDir, os.ModePerm); err != nil {
		return err
	}

	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}

	for _, file := range files {
		localFile := filepath.Join(targetDir, file)
		if _, err := os.Stat(localFile); os.IsNotExist(err) {
			log.Printf("Downloading %s...\n", file)
			if err := bp.DownloadFile(localFile, baseURL+file); err != nil {
				return err
			}
			log.Printf("Downloaded %s\n", file)

			if err := bp.UnzipFile(localFile, targetDir); err != nil {
				return err
			}
		} else {
			log.Printf("%s already exists, skipping download.\n", file)
		}
	}
	return nil
}

// UnpackMNIST processes MNIST data into images and labels.
func UnpackMNIST(imageFile, labelFile, outputDir string) error {
	imgFile, err := os.Open(imageFile)
	if err != nil {
		return fmt.Errorf("failed to open image file: %w", err)
	}
	defer imgFile.Close()

	lblFile, err := os.Open(labelFile)
	if err != nil {
		return fmt.Errorf("failed to open label file: %w", err)
	}
	defer lblFile.Close()

	var imgHeader [16]byte
	if _, err := imgFile.Read(imgHeader[:]); err != nil {
		return fmt.Errorf("failed to read image header: %w", err)
	}

	var lblHeader [8]byte
	if _, err := lblFile.Read(lblHeader[:]); err != nil {
		return fmt.Errorf("failed to read label header: %w", err)
	}

	numImages := binary.BigEndian.Uint32(imgHeader[4:8])
	imgRows := binary.BigEndian.Uint32(imgHeader[8:12])
	imgCols := binary.BigEndian.Uint32(imgHeader[12:16])

	numLabels := binary.BigEndian.Uint32(lblHeader[4:8])
	if numImages != numLabels {
		return fmt.Errorf("image and label count mismatch: %d images, %d labels", numImages, numLabels)
	}

	log.Printf("Processing %d images (%dx%d)...", numImages, imgRows, imgCols)

	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		return err
	}

	labelMap := make(map[string]int)
	imgSize := int(imgRows * imgCols)

	for i := 0; i < int(numImages); i++ {
		imgData := make([]byte, imgSize)
		if _, err := imgFile.Read(imgData); err != nil {
			return fmt.Errorf("failed to read image %d: %w", i, err)
		}

		var label byte
		if err := binary.Read(lblFile, binary.BigEndian, &label); err != nil {
			return fmt.Errorf("failed to read label %d: %w", i, err)
		}

		img := image.NewGray(image.Rect(0, 0, int(imgCols), int(imgRows)))
		copy(img.Pix, imgData)

		imgFilename := fmt.Sprintf("img_%05d.png", i)
		imgPath := filepath.Join(outputDir, imgFilename)

		imgOut, err := os.Create(imgPath)
		if err != nil {
			return fmt.Errorf("failed to create image file %s: %w", imgPath, err)
		}

		if err := png.Encode(imgOut, img); err != nil {
			imgOut.Close()
			return fmt.Errorf("failed to encode image %s: %w", imgPath, err)
		}
		imgOut.Close()

		labelMap[imgFilename] = int(label)

		if i%1000 == 0 {
			log.Printf("Processed %d/%d images...", i, numImages)
		}
	}

	labelMapPath := filepath.Join(outputDir, "labels.json")
	labelFileOut, err := os.Create(labelMapPath)
	if err != nil {
		return fmt.Errorf("failed to create label map file: %w", err)
	}
	defer labelFileOut.Close()

	encoder := json.NewEncoder(labelFileOut)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(labelMap); err != nil {
		return fmt.Errorf("failed to encode label map: %w", err)
	}

	log.Println("MNIST unpacked successfully.")
	return nil
}

// TrainOnMNIST trains the neural network using the MNIST dataset.
func TrainOnMNIST(bp *blueprint.Blueprint, mnistOutputDir string) error {
	rand.Seed(time.Now().UnixNano())

	// Load the label map
	labelMapPath := filepath.Join(mnistOutputDir, "labels.json")
	labelMapFile, err := os.Open(labelMapPath)
	if err != nil {
		return fmt.Errorf("failed to open label map file: %w", err)
	}
	defer labelMapFile.Close()

	var labelMap map[string]int
	if err := json.NewDecoder(labelMapFile).Decode(&labelMap); err != nil {
		return fmt.Errorf("failed to decode label map: %w", err)
	}

	// Create sessions
	var sessions []blueprint.Session
	var inputSize int
	for imgFilename, label := range labelMap {
		imgPath := filepath.Join(mnistOutputDir, imgFilename)
		imgFile, err := os.Open(imgPath)
		if err != nil {
			return fmt.Errorf("failed to open image file %s: %w", imgPath, err)
		}

		img, err := png.Decode(imgFile)
		imgFile.Close()
		if err != nil {
			return fmt.Errorf("failed to decode image %s: %w", imgPath, err)
		}

		grayImg := img.(*image.Gray)
		inputVars := make(map[int]float64)
		for i, pixel := range grayImg.Pix {
			inputVars[i+1] = float64(pixel) / 255.0 // Normalize [0, 1]
		}
		inputSize = len(grayImg.Pix) // 784 for MNIST

		// One-hot expected output
		expectedOutput := make(map[int]float64)
		for digit := 0; digit < 10; digit++ {
			outNodeID := 80001 + digit
			if digit == label {
				expectedOutput[outNodeID] = 1.0
			} else {
				expectedOutput[outNodeID] = 0.0
			}
		}

		sessions = append(sessions, blueprint.Session{
			InputVariables: inputVars,
			ExpectedOutput: expectedOutput,
			Timesteps:      1,
		})
	}

	// Define input and output nodes
	inputNodes := make([]int, inputSize)
	for i := 0; i < inputSize; i++ {
		inputNodes[i] = i + 1
	}
	outputNodes := []int{80001, 80002, 80003, 80004, 80005, 80006, 80007, 80008, 80009, 80010}

	bp.AddInputNodes(inputNodes)
	bp.AddOutputNodes(outputNodes)

	// Initialize input neurons
	for _, id := range inputNodes {
		bp.Neurons[id] = &blueprint.Neuron{
			ID:   id,
			Type: "input",
		}
	}

	// Initialize output neurons as linear
	for _, outID := range outputNodes {
		bp.Neurons[outID] = &blueprint.Neuron{
			ID:          outID,
			Type:        "output",
			Activation:  "linear",
			Connections: [][]float64{},
		}
	}

	// Parameters for NAS
	maxIterations := 10 // Adjust as needed
	forgivenessThreshold := 0.1
	neuronTypes := []string{"dense", "rnn", "cnn", "dropout", "attention"}
	weightUpdateIterations := 10

	fmt.Println("Training the model with ParallelSimpleNASWithRandomConnections...")
	bp.ParallelSimpleNASWithRandomConnections(sessions, maxIterations, forgivenessThreshold, neuronTypes, weightUpdateIterations)

	// Initialize PerformanceLogger
	logDir := filepath.Join(mnistDir, "log")
	logger, err := blueprint.NewPerformanceLogger(logDir)
	if err != nil {
		log.Fatalf("Failed to initialize PerformanceLogger: %v", err)
	}

	// Evaluate and log performance for all sessions
	if err := bp.EvaluateAndLogPerformance(sessions, logger); err != nil {
		log.Fatalf("Failed to evaluate and log performance: %v", err)
	}

	log.Println("Performance evaluation and logging completed successfully.")

	// After main training, try the targeted micro-refinement
	fmt.Println("Applying Targeted Micro Refinement...")
	bp.TargetedMicroRefinement(
		sessions,
		50,   // max refinement iterations
		20,   // sampleSubsetSize
		10,   // connectionTrialsPerSample
		30.0, // improvementThreshold for exact accuracy
	)

	// After refinement, try adding new connections using the improved multithreaded method
	fmt.Println("Trying to add new connections to improve accuracy...")
	bp.TryAddConnections(sessions, 50) // up to 50 unique connection attempts

	// Perform learning by processing one data item at a time
	fmt.Println("\nStarting LearnOneDataItemAtATime phase...")
	bp.LearnOneDataItemAtATime(sessions[:10], 10, neuronTypes, 5) // Adjust maxAttemptsPerSession as needed

	// Test the final model on a few samples
	fmt.Println("\nTesting the final model (raw predictions):")
	for i, session := range sessions[:10] {
		bp.RunNetwork(session.InputVariables, session.Timesteps)
		predictedOutput := bp.GetOutputs()

		// Apply softmax
		probs := softmaxMap(predictedOutput)
		predClass := argmaxMap(probs)
		expClass := argmaxMap(session.ExpectedOutput)

		fmt.Printf("Test %d: Expected: %d, Predicted: %d, Probabilities: %v\n", i+1, expClass, predClass, probs)
	}

	// Ensure the models directory exists
	completeModelDir := filepath.Join(mnistDir, modelDir)
	if err := os.MkdirAll(completeModelDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	// Save the model to mnist/models/mnist_model.json
	modelPath := filepath.Join(completeModelDir, modelName)
	if err := bp.SaveToJSON(modelPath); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	fmt.Printf("\nTraining complete. Model saved to %s\n", modelPath)
	return nil
}

// softmaxMap applies softmax to the values in a map and returns a new map with probabilities.
func softmaxMap(m map[int]float64) map[int]float64 {
	var sumExp float64
	for _, v := range m {
		sumExp += math.Exp(v)
	}
	probs := make(map[int]float64)
	for k, v := range m {
		probs[k] = math.Exp(v) / sumExp
	}
	return probs
}

// argmaxMap returns the key of the maximum value in the map.
func argmaxMap(m map[int]float64) int {
	var maxKey int
	var maxVal float64 = -math.MaxFloat64
	for k, v := range m {
		if v > maxVal {
			maxVal = v
			maxKey = k
		}
	}
	// Convert from neuron ID to class index (0-9)
	return maxKey - 80001
}
