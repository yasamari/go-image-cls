package cls

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type ModelConfig struct {
	ModelPath string
	InputName string
	Outputs   []OutputConfig
	Shape     Shape
}

type OutputConfig struct {
	Name   string
	Dim    int
	Labels []string
}

type PreprocessConfig struct {
	ColorFormatFunc        ColorFormatFunc
	ProcessImageFuncs      []ProcessImageFunc
	ProcessFloatImageFuncs []ProcessFloatImageFunc
}

type SessionConfig struct {
	Cuda        CudaConfig
	TensortRT   TensorRTConfig
	CoreML      CoreMLConfig
	DirectML    DirectMLConfig
	CpuMemArena *bool
	MemPattern  *bool

	InterOpNumThreads int
	IntraOpNumThreads int
}

type CudaConfig struct {
	Enabled bool
	// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
	Options map[string]string
}

type TensorRTConfig struct {
	Enabled bool
	// https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options
	Options map[string]string
}

type CoreMLConfig struct {
	Enabled bool
	//https://github.com/microsoft/onnxruntime/blob/291a5352b27ded5714e5748b381f2efb88f28fb9/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h#L12
	Flags uint32
}

type DirectMLConfig struct {
	Enabled  bool
	DeviceID int
}

type OpenVINOConfig struct {
	Enabled bool
	//https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
	Options map[string]string
}

func (s *SessionConfig) createSessionOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("Error create session options: %w", err)
	}

	if err := opts.SetInterOpNumThreads(s.InterOpNumThreads); err != nil {
		return nil, fmt.Errorf("Error set inter op num threads: %w", err)
	}

	if err := opts.SetIntraOpNumThreads(s.IntraOpNumThreads); err != nil {
		return nil, fmt.Errorf("Error set intra op num threads: %w", err)
	}

	if s.CpuMemArena != nil {
		if err := opts.SetCpuMemArena(*s.CpuMemArena); err != nil {
			return nil, fmt.Errorf("Error set cpu men arena: %w", err)
		}
	}
	if s.MemPattern != nil {
		if err := opts.SetMemPattern(*s.MemPattern); err != nil {
			return nil, fmt.Errorf("Error set mem pattern: %w", err)
		}
	}

	if s.Cuda.Enabled {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("Error create cuda execution provider: %w", err)
		}
		defer cudaOpts.Destroy()
		if err := cudaOpts.Update(s.Cuda.Options); err != nil {
			return nil, fmt.Errorf("Error update cuda options: %w", err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			return nil, fmt.Errorf("Error append cuda execution provider: %w", err)
		}
	}
	if s.TensortRT.Enabled {
		tensorRTOpts, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("Error create tensor rt execution provider: %w", err)
		}
		defer tensorRTOpts.Destroy()
		if err := tensorRTOpts.Update(s.TensortRT.Options); err != nil {
			return nil, fmt.Errorf("Error update tensorRT options: %w", err)
		}
		if err := opts.AppendExecutionProviderTensorRT(tensorRTOpts); err != nil {
			return nil, fmt.Errorf("Error append tensorRT excution provider: %w", err)
		}
	}
	if s.CoreML.Enabled {
		if err := opts.AppendExecutionProviderCoreML(s.CoreML.Flags); err != nil {
			return nil, fmt.Errorf("Error append coreML execution provider: %w", err)
		}
	}
	if s.DirectML.Enabled {
		if err := opts.AppendExecutionProviderDirectML(s.DirectML.DeviceID); err != nil {
			return nil, fmt.Errorf("Error append directml execution provider: %w", err)
		}
	}
	return opts, nil
}
