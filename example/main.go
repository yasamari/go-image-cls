package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"

	"github.com/nfnt/resize"
	cls "github.com/yasamari/go-image-cls"
)

const dirPath = "images"

func main() {
	cls.InitOnnxRuntime("onnxruntime-linux-x64-1.19.0/lib/libonnxruntime.so.1.19.0")

	c, err := loadClassifier()
	if err != nil {
		panic(err)
	}

	images, imagePaths, err := loadImages()
	if err != nil {
		panic(err)
	}

	outputs, err := c.Run(images)
	if err != nil {
		panic(err)
	}

	labelProbs := outputs[0].Label()

	for i, l := range labelProbs {
		label, prob := l.Get()
		fmt.Println(imagePaths[i], label, prob)
	}
}

func loadClassifier() (*cls.Classifier, error) {
	size := 384

	//https://hf.co/deepghs/monochrome_detect
	modelConf := &cls.ModelConfig{
		ModelPath: "model.onnx",
		InputName: "input",
		Outputs: []cls.OutputConfig{
			{
				Name: "output",
				Dim:  2,
				Labels: []string{
					"monochrome",
					"normal",
				},
			},
		},
		Shape: cls.NewBCHW(size),
	}

	preprocessConf := &cls.PreprocessConfig{
		ColorFormatFunc: cls.ColorFormatRGB,
		ProcessImageFuncs: []cls.ProcessImageFunc{
			cls.ResizeImage(size, resize.Bilinear),
		},
		ProcessFloatImageFuncs: []cls.ProcessFloatImageFunc{
			cls.Rescale(),
			cls.Normalize(0.5, 0.5),
		},
	}

	sessionConf := &cls.SessionConfig{
		InterOpNumThreads: 4,
		IntraOpNumThreads: 4,
	}

	c, err := cls.NewClassifier(modelConf, preprocessConf, sessionConf)

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
