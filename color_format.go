package cls

type ColorFormatFunc func(r, g, b float32) (first, second, third float32)

func ColorFormatRGB(r, g, b float32) (first, second, third float32) {
	return r, g, b
}

func ColorFormatBGR(r, g, b float32) (first, second, third float32) {
	return b, g, r
}
