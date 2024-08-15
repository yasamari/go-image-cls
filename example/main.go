package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
	cls "github.com/yasamari/go-image-cls"
)

const dirPath = "images"

func main() {
	cls.InitOnnxRuntime("/usr/lib/libonnxruntime.so")

	c, err := loadClassifier()
	if err != nil {
		panic(err)
	}

	images, imagePaths, err := loadImages()
	if err != nil {
		panic(err)
	}

	topLabels, err := c.Run(images)
	if err != nil {
		panic(err)
	}

	for i, label := range topLabels {
		fmt.Println(imagePaths[i], label)
	}
}

func loadClassifier() (*cls.Classifier, error) {
	opt, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opt.Destroy()

	cudaOpt, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return nil, err
	}
	defer cudaOpt.Destroy()

	err = opt.AppendExecutionProviderCUDA(cudaOpt)
	if err != nil {
		return nil, err
	}

	c, err := cls.NewClassifier(&cls.ClassifierConfig{
		ModelPath:      "model.onnx",
		InputName:      "input",
		OutputName:     "output",
		Size:           384,
		OutputDim:      2,
		SessionOptions: opt,
		Labels: []string{
			"monochrome",
			"normal",
		},
		Shape: cls.ShapeBCHW,
		Preprocess: cls.PreprocessConfig{
			ColorFormat: cls.FormatRGB,
			Normalize:   true,
			Padding:     false,
		},
	})
	if err != nil {
		return nil, err
	}
	return c, err
}

func loadImages() ([]image.Image, []string, error) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return nil, nil, err
	}

	var images []image.Image
	var imagePaths []string

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		imagePath := filepath.Join(dirPath, entry.Name())

		file, err := os.Open(imagePath)
		if err != nil {
			return nil, nil, err
		}
		img, err := jpeg.Decode(file)
		file.Close()
		if err != nil {
			return nil, nil, err
		}
		images = append(images, img)
		imagePaths = append(imagePaths, imagePath)
	}

	return images, imagePaths, nil
}
