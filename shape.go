package cls

type Shape interface {
	Index(batchIdx, x, y int) (first, second, third int)
	Get(batch int) []int64
	Size() int
}

type bhwc struct {
	size int
}

func NewBHWC(size int) Shape {
	return &bhwc{
		size: size,
	}
}

func (b *bhwc) Index(batchIdx, x, y int) (first, second, third int) {
	index := (batchIdx*b.size*b.size + y*b.size + x) * 3
	return index, index + 1, index + 2
}

func (b *bhwc) Get(batch int) []int64 {
	return []int64{int64(batch), int64(b.size), int64(b.size), 3}
}

func (b *bhwc) Size() int {
	return b.size
}

type bchw struct {
	size int
}

func NewBCHW(size int) Shape {
	return &bchw{
		size: size,
	}
}

func (b *bchw) Index(batchIdx, x, y int) (first, second, third int) {
	size := b.size
	first = batchIdx*3*size*size + 0*size*size + y*size + x
	second = batchIdx*3*size*size + 1*size*size + y*size + x
	third = batchIdx*3*size*size + 2*size*size + y*size + x
	return first, second, third
}

func (b *bchw) Get(batch int) []int64 {
	return []int64{int64(batch), 3, int64(b.size), int64(b.size)}
}

func (b *bchw) Size() int {
	return b.size
}
