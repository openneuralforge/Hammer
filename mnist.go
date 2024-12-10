package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"math/rand"
	"time"
	"blueprint"
)

const (
	baseURL  = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	mnistDir = "mnist"
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

	// Train the model using SimpleNASWithoutCrossover
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

		// Flatten the image into input variables
		inputVars := make(map[int]float64)
		grayImg := img.(*image.Gray)
		for i, pixel := range grayImg.Pix {
			inputVars[i+1] = float64(pixel) / 255.0 // Normalize to [0, 1]
		}

		// Add the session
		sessions = append(sessions, blueprint.Session{
			InputVariables: inputVars,
			ExpectedOutput: map[int]float64{
				1: float64(label), // Assume a single output neuron
			},
			Timesteps: 1,
		})
	}

	// Define input and output nodes
	inputNodes := make([]int, len(sessions[0].InputVariables))
	for i := range inputNodes {
		inputNodes[i] = i + 1
	}
	outputNodes := []int{1}

	bp.AddInputNodes(inputNodes)
	bp.AddOutputNodes(outputNodes)

	// Initialize neurons
	for _, id := range inputNodes {
		bp.Neurons[id] = &blueprint.Neuron{
			ID:   id,
			Type: "input",
		}
	}
	bp.Neurons[1] = &blueprint.Neuron{
		ID:          1,
		Type:        "output",
		Activation:  "softmax",
		Connections: [][]float64{},
	}

	// Set parameters for SimpleNASWithoutCrossover
	maxIterations := 65000
	forgivenessThreshold := 0.1 // 10%
	neuronTypes := []string{
		"dense",
		"rnn",
		//"lstm",
		"cnn",
		"dropout",
		//"batch_norm",
		"attention",
		//"nca",
	}
	//metrics := []string{"exact","generous","forgiveness"}
	weightUpdateIterations := 10 // Number of weight update steps per NAS iteration

	// Perform SimpleNASWithoutCrossover
	fmt.Println("Training the model with ParallelSimpleNASWithRandomConnections...")
	//bp.SimpleNASWithRandomConnections(sessions, maxIterations, forgivenessThreshold, neuronTypes, metrics)
	//bp.SimpleNASWithRandomConnections(sessions, maxIterations, forgivenessThreshold, neuronTypes, weightUpdateIterations)
	bp.ParallelSimpleNASWithRandomConnections(sessions, maxIterations, forgivenessThreshold, neuronTypes, weightUpdateIterations)
	
	// Test the final model
	fmt.Println("Testing the final model:")
	for _, session := range sessions[:10] { // Test on the first 10 sessions
		bp.RunNetwork(session.InputVariables, session.Timesteps)
		predictedOutput := bp.GetOutputs()
		fmt.Printf("Input: %v, Expected Output: %v, Predicted Output: %v\n", session.InputVariables, session.ExpectedOutput, predictedOutput)
	}

	// Save the model
	if err := bp.SaveToJSON(filepath.Join(mnistOutputDir, "mnist_model.json")); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	fmt.Println("Training complete. Model saved to mnist_model.json")
	return nil
}