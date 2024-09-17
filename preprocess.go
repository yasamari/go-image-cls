package cls

import (
	"image"
)

func (c *Classifier) preprocessImages(images []image.Image) ([]float32, error) {
	var processedImages []image.Image
	for _, img := range images {
		for _, fn := range c.preprocessConf.ProcessImageFuncs {
			img = fn(img)
		}
		processedImages = append(processedImages, img)
	}

	size := c.modelConf.Shape.Size()

	batch := len(images)
	result := make([]float32, batch*size*size*3)

	for n, img := range processedImages {
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				r, g, b, _ := img.At(x, y).RGBA()

				rFloat := float32(r / 257)
				gFloat := float32(g / 257)
				bFloat := float32(b / 257)

				for _, fn := range c.preprocessConf.ProcessFloatImageFuncs {
					rFloat = fn(rFloat)
					gFloat = fn(gFloat)
					bFloat = fn(bFloat)
				}

				first, second, third := c.preprocessConf.ColorFormatFunc(rFloat, gFloat, bFloat)
				firstIdx, secondIdx, thirdIdx := c.modelConf.Shape.Index(n, x, y)

				result[firstIdx] = first
				result[secondIdx] = second
				result[thirdIdx] = third
			}
		}
	}
	return result, nil
}
