package cls

import ort "github.com/yalue/onnxruntime_go"

func InitOnnxRuntime(libPath string) {
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		return
	}
}

func DestroyEnvironment() error {
	return ort.DestroyEnvironment()
}
