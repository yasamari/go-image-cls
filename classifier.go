package cls

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
)

type Classifier struct {
	conf                  *Config
	session               *ort.DynamicAdvancedSession
	inputInfo, outputInfo []ort.InputOutputInfo
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

func (o *Output) Labels() [][]*LabelProb {
	var result [][]*LabelProb
	for _, probs := range o.data {
		var labelProbs []*LabelProb
		for idx, prob := range probs {
			label := o.labels[idx]
			labelProbs = append(labelProbs, &LabelProb{
				label: label,
				prob:  prob,
			})
		}
		result = append(result, labelProbs)
	}
	return result
}

func (o *Output) TopLabels() []*LabelProb {
	batchLabels := o.Labels()
	var result []*LabelProb
	for _, labels := range batchLabels {
		top := labels[0]
		for _, l := range labels {
			if l.prob > top.prob {
				top = l
			}
		}
		result = append(result, top)
	}
	return result
}

func NewClassifier(conf *Config, sessionConf *SessionConfig) (*Classifier, error) {
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(conf.ModelPath)

	var inputNames []string
	for _, info := range inputInfo {
		inputNames = append(inputNames, info.Name)
	}

	var outputNames []string
	for _, info := range outputInfo {
		outputNames = append(outputNames, info.Name)
	}

	opts, err := sessionConf.createSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("Error create session options: %w", err)
	}
	defer opts.Destroy()

	session, err := ort.NewDynamicAdvancedSession(conf.ModelPath, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("Error load model: %w", err)
	}
	return &Classifier{
		conf:       conf,
		session:    session,
		inputInfo:  inputInfo,
		outputInfo: outputInfo,
	}, nil
}

func (c *Classifier) Close() {
	c.session.Destroy()
}

func (c *Classifier) Run(images []image.Image) ([]*Output, error) {
	outputs, err := c.RunRaw(images)
	if err != nil {
		return nil, err
	}

	var result []*Output
	for idx, o := range outputs {
		labels, ok := c.conf.OutputLabels[idx]
		if !ok {
			labels = nil
		}
		result = append(result, &Output{
			data:   o,
			labels: labels,
		})
	}

	return result, nil
}

func (c *Classifier) RunRaw(images []image.Image) ([][][]float32, error) {
	input, err := c.preprocessImages(images)
	if err != nil {
		return nil, err
	}

	batch := int64(len(images))

	inputShape := []int64{batch, 3, int64(c.conf.Height), int64(c.conf.Width)}
	if c.conf.Shape == BHWC {
		inputShape = []int64{batch, int64(c.conf.Height), int64(c.conf.Width), 3}
	}

	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("Error create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	var outputTensors []ort.ArbitraryTensor
	for _, output := range c.outputInfo {
		outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batch, output.Dimensions[1]))
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

	for _, outputValue := range outputTensors {
		tensor := outputValue.(*ort.Tensor[float32])
		var tensorResult [][]float32
		data := tensor.GetData()
		labelDim := int(tensor.GetShape()[1])

		for b := 0; b < int(batch); b++ {
			tensorResult = append(tensorResult, data[labelDim*b:labelDim*(b+1)])
		}
		result = append(result, tensorResult)
	}
	return result, nil
}
