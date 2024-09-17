package cls

import (
	"image"
	"image/color"
	"image/draw"

	"github.com/nfnt/resize"
)

type ProcessImageFunc func(img image.Image) image.Image

type ProcessFloatImageFunc func(rgb float32) float32

func ResizeWithPadding(size int, padColor color.Color, interp resize.InterpolationFunction) ProcessImageFunc {
	return func(img image.Image) image.Image {
		img = resize.Thumbnail(uint(size), uint(size), img, resize.Bilinear)
		canvas := image.NewRGBA(image.Rect(0, 0, size, size))
		draw.Draw(canvas, canvas.Bounds(), image.NewUniform(padColor), image.Point{}, draw.Over)

		offsetX := (size - img.Bounds().Dx()) / 2
		offsetY := (size - img.Bounds().Dy()) / 2
		draw.Draw(canvas, img.Bounds().Add(image.Point{offsetX, offsetY}), img, image.Point{}, draw.Over)
		return canvas
	}
}

func ResizeImage(size int, interp resize.InterpolationFunction) ProcessImageFunc {
	return func(img image.Image) image.Image {
		return resize.Resize(uint(size), uint(size), img, interp)
	}
}

func Rescale() ProcessFloatImageFunc {
	return func(rgb float32) float32 {
		return rgb / 255.0
	}
}

func Normalize(mean, std float32) ProcessFloatImageFunc {
	return func(rgb float32) float32 {
		return (rgb - mean) / std
	}
}
