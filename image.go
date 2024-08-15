package cls

import (
	"errors"
	"fmt"
	"image"
)

type Shape int

const (
	//(batch, height, width, channel)
	ShapeBHWC Shape = iota
	// (batch, channel, height, width)
	ShapeBCHW
)

type ColorFormat int

const (
	FormatBGR ColorFormat = iota
	FormatRGB
)

func imageToFloat32(images []image.Image, shape Shape, color ColorFormat, normalize bool) ([]float32, error) {
	firstBounds := images[0].Bounds()
	width, height := firstBounds.Dx(), firstBounds.Dy()
	for _, img := range images[:1] {
		if img.Bounds().Dx() != width || img.Bounds().Dy() != height {
			return nil, fmt.Errorf("all images must have the same dimensions")
		}
	}

	batch := len(images)
	result := make([]float32, batch*height*width*3)

	for n, img := range images {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x, y).RGBA()

				rFloat := float32(r >> 8)
				gFloat := float32(g >> 8)
				bFloat := float32(b >> 8)

				if normalize {
					rFloat = rFloat / 255.0
					gFloat = gFloat / 255.0
					bFloat = bFloat / 255.0
				}

				first, second, third := rFloat, gFloat, bFloat
				if color == FormatBGR {
					first, second, third = bFloat, gFloat, rFloat
				}

				switch shape {
				case ShapeBCHW:
					result[n*3*height*width+0*height*width+y*width+x] = first
					result[n*3*height*width+1*height*width+y*width+x] = second
					result[n*3*height*width+2*height*width+y*width+x] = third
				case ShapeBHWC:
					index := (n*height*width + y*width + x) * 3
					result[index] = first
					result[index+1] = second
					result[index+2] = third
				default:
					return nil, errors.New("invalid shape")
				}
			}
		}
	}
	return result, nil
}
