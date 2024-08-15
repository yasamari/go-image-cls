package cls

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

type ClassifierConfig struct {
	ModelPath             string
	InputName, OutputName string
	Size                  int
	OutputDim             int
	SessionOptions        *ort.SessionOptions
	Labels                []string
	Shape
	Preprocess PreprocessConfig
}

type PreprocessConfig struct {
	ColorFormat
	Normalize    bool
	Padding      bool
	PaddingColor color.Gray16
}

type Classifier struct {
	conf    *ClassifierConfig
	session *ort.DynamicAdvancedSession
}

func NewClassifier(conf *ClassifierConfig) (*Classifier, error) {
	session, err := ort.NewDynamicAdvancedSession(conf.ModelPath, []string{conf.InputName}, []string{conf.OutputName}, conf.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("Error load model: %w", err)
	}
	return &Classifier{
		conf:    conf,
		session: session,
	}, nil
}

func (c *Classifier) preprocessImages(images []image.Image) ([]float32, error) {
	conf := &c.conf.Preprocess
	size := c.conf.Size
	var resizedImages []image.Image
	for _, img := range images {
		if conf.Padding {
			img = resize.Thumbnail(uint(size), uint(size), img, resize.Bicubic)
			canvas := image.NewRGBA(image.Rect(0, 0, size, size))
			draw.Draw(canvas, canvas.Bounds(), image.NewUniform(conf.PaddingColor), image.Point{}, draw.Over)

			offsetX := (size - img.Bounds().Dx()) / 2
			offsetY := (size - img.Bounds().Dy()) / 2
			draw.Draw(canvas, img.Bounds().Add(image.Point{offsetX, offsetY}), img, image.Point{}, draw.Over)
			img = canvas
		} else {
			img = resize.Resize(uint(size), uint(size), img, resize.Bilinear)
		}
		resizedImages = append(resizedImages, img)
	}
	return imageToFloat32(resizedImages, c.conf.Shape, conf.ColorFormat, conf.Normalize)
}

func (c *Classifier) Run(images []image.Image) ([]string, error) {
	output, err := c.RunRaw(images)
	if err != nil {
		return nil, err
	}

	var result []string

	for _, probs := range output {
		topLabel := c.conf.Labels[0]
		var topProb float32 = 0.0
		for i, label := range c.conf.Labels {
			if probs[i] > topProb {
				topLabel = label
				topProb = probs[i]
			}
		}
		result = append(result, topLabel)
	}
	return result, nil
}

func (c *Classifier) RunRaw(images []image.Image) ([][]float32, error) {
	input, err := c.preprocessImages(images)
	if err != nil {
		return nil, err
	}

	size := int64(c.conf.Size)
	batch := int64(len(images))

	var inputShape ort.Shape

	switch c.conf.Shape {
	case ShapeBCHW:
		inputShape = ort.NewShape(batch, 3, size, size)
	case ShapeBHWC:
		inputShape = ort.NewShape(batch, size, size, 3)
	default:
		return nil, errors.New("invalid shape")
	}

	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("Error create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(batch, int64(c.conf.OutputDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("Error create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = c.session.Run([]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("Error run inference: %w", err)
	}

	output := outputTensor.GetData()
	var result [][]float32

	for b := 0; b < int(batch); b++ {
		result = append(result, output[c.conf.OutputDim*b:c.conf.OutputDim*(b+1)])
	}

	return result, nil
}
