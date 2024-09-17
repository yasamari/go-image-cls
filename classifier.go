package cls

import (
	"errors"
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
)

type Classifier struct {
	modelConf      *ModelConfig
	preprocessConf *PreprocessConfig
	session        *ort.DynamicAdvancedSession
	opts           *ort.SessionOptions
}

type LabelProb struct {
	label string
	prob  float32
}

func (l *LabelProb) Get() (string, float32) {
	return l.label, l.prob
}

type Output struct {
	data   [][]float32
	labels []string
}

func (o *Output) Raw() [][]float32 {
	return o.data
}

func (o *Output) Label() []*LabelProb {
	var result []*LabelProb
	for _, probs := range o.data {
		topLabel := o.labels[0]
		var topProb float32
		for i, label := range o.labels {
			if probs[i] > topProb {
				topLabel = label
				topProb = probs[i]
			}
		}
		result = append(result, &LabelProb{
			label: topLabel,
			prob:  topProb,
		})
	}
	return result
}

func NewClassifier(modelConf *ModelConfig, preprocessConf *PreprocessConfig, sessionConf *SessionConfig) (*Classifier, error) {
	var outputNames []string
	for _, output := range modelConf.Outputs {
		outputNames = append(outputNames, output.Name)
	}

	opts, err := sessionConf.createSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("Error create session options: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelConf.ModelPath, []string{modelConf.InputName}, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("Error load model: %w", err)
	}
	return &Classifier{
		modelConf:      modelConf,
		preprocessConf: preprocessConf,
		session:        session,
		opts:           opts,
	}, nil
}

func (c *Classifier) Close() {
	c.session.Destroy()
	c.opts.Destroy()
}

func (c *Classifier) Run(images []image.Image) ([]*Output, error) {
	outputs, err := c.RunRaw(images)
	if err != nil {
		return nil, err
	}

	var result []*Output
	for idx, outputConf := range c.modelConf.Outputs {
		result = append(result, &Output{
			data:   outputs[idx],
			labels: outputConf.Labels,
		})
	}

	return result, nil
}

func (c *Classifier) RunRaw(images []image.Image) ([][][]float32, error) {
	input, err := c.preprocessImages(images)
	if err != nil {
		return nil, err
	}

	batch := len(images)

	inputShape := c.modelConf.Shape.Get(batch)

	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("Error create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	var outputTensors []ort.ArbitraryTensor
	for _, output := range c.modelConf.Outputs {
		outputShape := ort.NewShape(int64(batch), int64(output.Dim))
		outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return nil, fmt.Errorf("Error create output tensor: %w", err)
		}
		defer outputTensor.Destroy()
		outputTensors = append(outputTensors, outputTensor)
	}

	err = c.session.Run([]ort.ArbitraryTensor{inputTensor}, outputTensors)
	if err != nil {
		return nil, fmt.Errorf("Error run inference: %w", err)
	}

	var result [][][]float32

	for idx, output := range c.modelConf.Outputs {
		var tensorResult [][]float32
		outputTensor, ok := outputTensors[idx].(*ort.Tensor[float32])
		if !ok {
			return nil, errors.New("assert ort.ArbitraryTensor not ok")
		}
		data := outputTensor.GetData()
		for b := 0; b < int(batch); b++ {
			tensorResult = append(tensorResult, data[output.Dim*b:output.Dim*(b+1)])
		}
		result = append(result, tensorResult)
	}
	return result, nil
}
