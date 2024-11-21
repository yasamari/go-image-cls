package cls

import (
	"image"
)

func (c *Classifier) preprocessImages(images []image.Image) ([]float32, error) {
	var processedImages []image.Image
	for _, img := range images {
		for _, fn := range c.conf.ProcessImageFuncs {
			img = fn(img)
		}
		processedImages = append(processedImages, img)
	}

	bounds := processedImages[0].Bounds()
	height, width := bounds.Dy(), bounds.Dx()

	batch := len(images)
	result := make([]float32, batch*height*width*3)

	for n, img := range processedImages {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x, y).RGBA()

				rf := float32(r / 257.0)
				gf := float32(g / 257.0)
				bf := float32(b / 257.0)

				for _, fn := range c.conf.ProcessFloatImageFuncs {
					rf = fn(rf)
					gf = fn(gf)
					bf = fn(bf)
				}

				first, second, third := c.conf.ColorFormatFunc(rf, gf, bf)

				if c.conf.Shape == BCHW {
					result[n*3*height*width+0*height*width+y*width+x] = first
					result[n*3*height*width+1*height*width+y*width+x] = second
					result[n*3*height*width+2*height*width+y*width+x] = third
				} else {
					idx := (n*height*width + y*width + x) * 3
					result[idx] = first
					result[idx+1] = second
					result[idx+2] = third
				}
			}
		}
	}
	return result, nil
}
