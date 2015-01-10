package godnn

import (
	"errors"
	"github.com/gonum/blas"
	"math"
)

type ConvolutionLayer struct {
	BaseLayer
	NumOutputs   int
	NumGroups    int
	KernelHeight int
	KernelWidth  int
	PadHeight    int
	PadWidth     int
	StrideHeight int
	StrideWidth  int
	IncludeBias  bool

	bottomDim      *BlobPoint
	outputChannels int
	weightParams   *Blob
	biasParams     *Blob
	biasMultiplier *Blob
	heightTop      int
	widthTop       int
	m              int
	k              int
	n              int
	weightOffset   int
	colOffset      int
	topOffset      int
	colBuffer      *Blob
}

var _ = Layer(new(ConvolutionLayer))

func (l *ConvolutionLayer) Setup(d *LayerData) error {
	if len(l.TopNames) != len(l.BottomNames) {
		return errors.New("number of bottom and top layers needs to be the same")
	}

	l.bottomDim = &d.Bottom[0].Dim
	if l.bottomDim.Channel%l.NumGroups != 0 {
		return errors.New("number of channels needs to be multiple of number of groups")
	}
	if l.NumOutputs%l.NumGroups != 0 {
		return errors.New("number of outputs needs to be multiple of number of groups")
	}
	for n := 1; n < len(d.Bottom); n++ {
		if *l.bottomDim != d.Bottom[n].Dim {
			return errors.New("all bottom channels must be the same size")
		}
	}
	l.outputChannels = l.bottomDim.Channel / l.NumGroups

	l.heightTop = (l.bottomDim.Height+2*l.PadHeight-l.KernelHeight)/l.StrideHeight + 1
	l.widthTop = (l.bottomDim.Width+2*l.PadWidth-l.KernelWidth)/l.StrideWidth + 1
	l.m = l.NumOutputs / l.NumGroups
	l.k = l.bottomDim.Channel * l.KernelHeight * l.KernelWidth / l.NumGroups
	l.n = l.heightTop * l.widthTop
	l.weightOffset = l.m * l.k
	l.colOffset = l.k * l.n
	l.topOffset = l.m * l.n

	l.weightParams = NewBlob(l.LayerName()+"_weight",
		&BlobPoint{l.NumOutputs, l.outputChannels, l.KernelHeight, l.KernelWidth})
	if l.IncludeBias {
		l.biasParams = NewBlob(l.LayerName()+"_bias",
			&BlobPoint{1, 1, 1, l.NumOutputs})
		l.biasMultiplier = NewBlob(l.LayerName()+"_biasMultiplier",
			&BlobPoint{1, 1, 1, l.heightTop * l.widthTop})
	}
	// TODO: fill weights and bias terms
	Set32(l.biasMultiplier.Data.MutableCpuValues(), 1)

	l.colBuffer = NewBlob(l.LayerName()+"_colBuffer",
		&BlobPoint{1, l.bottomDim.Channel * l.KernelHeight * l.KernelWidth, l.heightTop, l.widthTop})

	d.Top = make([]*Blob, len(l.TopNames))
	for n := 0; n < len(l.TopNames); n++ {
		d.Top[n] = NewBlob(l.TopNames[n],
			&BlobPoint{l.bottomDim.Batch, l.NumOutputs, l.heightTop, l.widthTop})
	}

	return nil
}

func (l *ConvolutionLayer) FeedForward(d *LayerData) float32 {
	for i, bottom := range d.Bottom {
		top := d.Top[i]
		bottomData := bottom.Data.CpuValues()
		topData := top.Data.MutableCpuValues()
		colData := l.colBuffer.Data.MutableCpuValues()
		weight := l.weightParams.Data.CpuValues()

		for n := 0; n < l.bottomDim.Batch; n++ {
			bottomSlice := Subslice32(bottomData, n, bottom.Dim.BatchSize())
			topSlice := Subslice32(topData, n, top.Dim.BatchSize())

			Im2col32(bottomSlice, l.bottomDim.Channel, l.bottomDim.Height, l.bottomDim.Width,
				l.KernelHeight, l.KernelWidth, l.PadHeight, l.PadWidth,
				l.StrideHeight, l.StrideWidth, colData)

			for g := 0; g < l.NumGroups; g++ {
				Gemm32(blas.NoTrans, blas.NoTrans, l.m, l.n, l.k,
					1, Subslice32(weight, g, l.weightOffset), Subslice32(colData, g, l.colOffset),
					0, Subslice32(topSlice, g, l.topOffset))
			}

			if l.IncludeBias {
				biasData := l.biasParams.Data.CpuValues()
				biasMultiplierData := l.biasMultiplier.Data.CpuValues()
				Gemm32(blas.NoTrans, blas.NoTrans, l.NumOutputs, l.n, 1,
					1, biasData, biasMultiplierData,
					1, topSlice)
			}
		}
	}
	return 0
}

func (l *ConvolutionLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	weight := l.weightParams.Data.CpuValues()
	weightDiff := l.weightParams.Diff.MutableCpuValues()
	Set32(weightDiff, 0)
	biasDiff := l.biasParams.Diff.MutableCpuValues()
	Set32(biasDiff, 0)
	biasMultiplier := l.biasMultiplier.Data.CpuValues()

	for i, top := range d.Top {
		bottom := d.Bottom[i]
		topDiff := top.Diff.CpuValues()
		bottomData := bottom.Data.CpuValues()
		bottomDiff := bottom.Diff.MutableCpuValues()
		colData := l.colBuffer.Data.MutableCpuValues()
		colDiff := l.colBuffer.Diff.MutableCpuValues()

		for n := 0; n < l.bottomDim.Batch; n++ {
			bottomDataSlice := Subslice32(bottomData, n, l.bottomDim.BatchSize())
			bottomDiffSlice := Subslice32(bottomDiff, n, l.bottomDim.BatchSize())

			Im2col32(bottomDataSlice, l.bottomDim.Channel, l.bottomDim.Height, l.bottomDim.Width,
				l.KernelHeight, l.KernelWidth, l.PadHeight, l.PadWidth,
				l.StrideHeight, l.StrideWidth, colData)

			for g := 0; g < l.NumGroups; g++ {
				Gemm32(blas.NoTrans, blas.NoTrans, l.k, l.n, l.m,
					1, Subslice32(weight, g, l.weightOffset), Subslice32(topDiff, g, l.topOffset),
					0, Subslice32(colDiff, g, l.colOffset))

				if paramPropagate {
					Gemm32(blas.NoTrans, blas.NoTrans, l.m, l.k, l.n,
						1, Subslice32(topDiff, g, l.topOffset), Subslice32(colData, g, l.colOffset),
						1, Subslice32(weightDiff, g, l.weightOffset))
				}
			}

			Col2im32(colDiff, l.bottomDim.Channel, l.bottomDim.Height, l.bottomDim.Width,
				l.KernelHeight, l.KernelWidth, l.PadHeight, l.PadWidth,
				l.StrideHeight, l.StrideWidth, bottomDiffSlice)
		}

		if paramPropagate && l.IncludeBias {
			for n := 0; n < l.bottomDim.Batch; n++ {
				topDiffSlice := Subslice32(topDiff, n, top.Dim.BatchSize())

				Gemv32(blas.NoTrans, l.NumOutputs, l.n,
					1, topDiffSlice, biasMultiplier, 1, biasDiff)
			}
		}
	}
}

func (l *ConvolutionLayer) Params() []*Blob {
	if l.IncludeBias {
		return []*Blob{l.weightParams, l.biasParams, l.biasMultiplier}
	}
	return []*Blob{l.weightParams}
}

func NewConvolutionLayer(baseLayer BaseLayer,
	numOutputs, numGroups, kernelHeight, kernelWidth,
	padHeight, padWidth, strideHeight, strideWidth int,
	includeBias bool) *ConvolutionLayer {
	return &ConvolutionLayer{
		BaseLayer:    baseLayer,
		NumOutputs:   numOutputs,
		NumGroups:    numGroups,
		KernelHeight: kernelHeight,
		KernelWidth:  kernelWidth,
		PadHeight:    padHeight,
		PadWidth:     padWidth,
		StrideHeight: strideHeight,
		StrideWidth:  strideWidth,
		IncludeBias:  includeBias,
	}
}

type PoolMethod int

const (
	PoolMethodAverage PoolMethod = iota
	PoolMethodMax
)

type PoolingLayer struct {
	BaseLayer
	Method       PoolMethod
	NumOutputs   int
	NumGroups    int
	KernelHeight int
	KernelWidth  int
	PadHeight    int
	PadWidth     int
	StrideHeight int
	StrideWidth  int

	bottomDim    *BlobPoint
	topDim       *BlobPoint
	pooledHeight int
	pooledWidth  int
}

var _ = Layer(new(PoolingLayer))

func (l *PoolingLayer) Setup(d *LayerData) error {
	err := l.checkNames(1, 2)
	if err != nil {
		return err
	}

	l.bottomDim = &d.Bottom[0].Dim
	l.pooledHeight = int(Ceil32(float32(l.bottomDim.Height+2*l.PadHeight-l.KernelHeight))/float32(l.StrideHeight)) + 1
	l.pooledWidth = int(Ceil32(float32(l.bottomDim.Width+2*l.PadWidth-l.KernelWidth))/float32(l.StrideWidth)) + 1
	if (l.pooledHeight-1)*l.StrideHeight >= l.bottomDim.Height+l.PadHeight {
		l.pooledHeight--
	}
	if (l.pooledWidth-1)*l.StrideWidth >= l.bottomDim.Width+l.PadWidth {
		l.pooledWidth--
	}

	d.Top = make([]*Blob, 2)
	d.Top[0] = NewBlob(l.TopNames[0], &BlobPoint{l.bottomDim.Batch, l.bottomDim.Channel,
		l.pooledHeight, l.pooledWidth})
	d.Top[1] = NewBlob(l.TopNames[1], &d.Top[0].Dim)
	l.topDim = &d.Top[0].Dim
	return nil
}

func (l *PoolingLayer) FeedForward(d *LayerData) float32 {
	bottomData := d.Bottom[0].Data.CpuValues()
	topData := d.Top[0].Data.MutableCpuValues()

	switch l.Method {
	case PoolMethodMax:
		topMask := d.Top[1].Data.MutableCpuValues()
		l.feedForwardMax(bottomData, topData, topMask)
	case PoolMethodAverage:
		l.feedForwardAverage(bottomData, topData)
	}
	return 0
}

func (l *PoolingLayer) feedForwardMax(bottomData, topData, topMask []float32) {
	Set32(topMask, -1)
	Set32(topData, float32(math.Inf(-1)))
	bottomSpatialSize := l.bottomDim.SpatialSize()
	topSpatialSize := l.topDim.SpatialSize()
	for n := 0; n < l.bottomDim.Batch; n++ {
		for c := 0; c < l.bottomDim.Channel; c++ {
			channelIndex := n*l.bottomDim.Channel + c
			bottomDataSlice := Subslice32(bottomData, channelIndex, bottomSpatialSize)
			topDataSlice := Subslice32(topData, channelIndex, topSpatialSize)
			topMaskSlice := Subslice32(topMask, channelIndex, topSpatialSize)

			for ph := 0; ph < l.pooledHeight; ph++ {
				for pw := 0; pw < l.pooledWidth; pw++ {
					hstart := ph*l.StrideHeight - l.PadHeight
					wstart := pw*l.StrideWidth - l.PadWidth
					hend := Min(hstart+l.KernelHeight, l.bottomDim.Height)
					wend := Min(wstart+l.KernelWidth, l.bottomDim.Width)
					hstart = Max(hstart, 0)
					wstart = Max(wstart, 0)
					poolIndex := ph*l.pooledWidth + pw
					for h := hstart; h < hend; h++ {
						for w := wstart; w < wend; w++ {
							index := h*l.bottomDim.Width + w
							if bottomDataSlice[index] > topDataSlice[poolIndex] {
								topDataSlice[poolIndex] = bottomDataSlice[index]
								topMaskSlice[poolIndex] = float32(index)
							}
						}
					}
				}
			}
		}
	}
}

func (l *PoolingLayer) feedForwardAverage(bottomData, topData []float32) {
	Set32(topData, 0)
	bottomSpatialSize := l.bottomDim.SpatialSize()
	topSpatialSize := l.topDim.SpatialSize()
	for n := 0; n < l.bottomDim.Batch; n++ {
		for c := 0; c < l.bottomDim.Channel; c++ {
			channelIndex := n*l.bottomDim.Channel + c
			bottomDataSlice := Subslice32(bottomData, channelIndex, bottomSpatialSize)
			topDataSlice := Subslice32(topData, channelIndex, topSpatialSize)

			for ph := 0; ph < l.pooledHeight; ph++ {
				for pw := 0; pw < l.pooledWidth; pw++ {
					hstart := ph*l.StrideHeight - l.PadHeight
					wstart := pw*l.StrideWidth - l.PadWidth
					hend := Min(hstart+l.KernelHeight, l.bottomDim.Height)
					wend := Min(wstart+l.KernelWidth, l.bottomDim.Width)
					hstart = Max(hstart, 0)
					wstart = Max(wstart, 0)
					hend = Min(hend, l.bottomDim.Height)
					wend = Min(wend, l.bottomDim.Width)
					poolIndex := ph*l.pooledWidth + pw
					for h := hstart; h < hend; h++ {
						for w := wstart; w < wend; w++ {
							index := h*l.bottomDim.Width + w
							topDataSlice[poolIndex] += bottomDataSlice[index]
						}
					}
					poolSize := (hend - hstart) * (wend - wstart)
					topDataSlice[poolIndex] /= float32(poolSize)
				}
			}
		}
	}
}

func (l *PoolingLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	topDiff := d.Top[0].Diff.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()

	switch l.Method {
	case PoolMethodMax:
		topMask := d.Top[1].Data.CpuValues()
		l.feedBackwardMax(topDiff, bottomDiff, topMask)
	case PoolMethodAverage:
		l.feedBackwardAverage(topDiff, bottomDiff)
	}
}

func (l *PoolingLayer) feedBackwardMax(topDiff, bottomDiff, topMask []float32) {
	topSpatialSize := l.topDim.SpatialSize()
	bottomSpatialSize := l.bottomDim.SpatialSize()
	for n := 0; n < l.bottomDim.Batch; n++ {
		for c := 0; c < l.bottomDim.Channel; c++ {
			channelIndex := n*l.bottomDim.Channel + c
			topDiffSlice := Subslice32(topDiff, channelIndex, topSpatialSize)
			topMaskSlice := Subslice32(topMask, channelIndex, topSpatialSize)
			bottomDiffSlice := Subslice32(bottomDiff, channelIndex, bottomSpatialSize)

			for ph := 0; ph < l.pooledHeight; ph++ {
				for pw := 0; pw < l.pooledWidth; pw++ {
					index := ph*l.pooledWidth + pw
					bottomIndex := int(topMaskSlice[index])
					bottomDiffSlice[bottomIndex] += topDiffSlice[index]
				}
			}
		}
	}
}

func (l *PoolingLayer) feedBackwardAverage(topDiff, bottomDiff []float32) {
	bottomSpatialSize := l.bottomDim.SpatialSize()
	topSpatialSize := l.topDim.SpatialSize()
	for n := 0; n < l.bottomDim.Batch; n++ {
		for c := 0; c < l.bottomDim.Channel; c++ {
			channelIndex := n*l.bottomDim.Channel + c
			topDiffSlice := Subslice32(topDiff, channelIndex, topSpatialSize)
			bottomDiffSlice := Subslice32(bottomDiff, channelIndex, bottomSpatialSize)

			for ph := 0; ph < l.pooledHeight; ph++ {
				for pw := 0; pw < l.pooledWidth; pw++ {
					hstart := ph*l.StrideHeight - l.PadHeight
					wstart := pw*l.StrideWidth - l.PadWidth
					hend := Min(hstart+l.KernelHeight, l.bottomDim.Height)
					wend := Min(wstart+l.KernelWidth, l.bottomDim.Width)
					hstart = Max(hstart, 0)
					wstart = Max(wstart, 0)
					hend = Min(hend, l.bottomDim.Height)
					wend = Min(wend, l.bottomDim.Width)
					poolIndex := ph*l.pooledWidth + pw
					poolSize := (hend - hstart) * (wend - wstart)
					for h := hstart; h < hend; h++ {
						for w := wstart; w < wend; w++ {
							bottomIndex := h*l.bottomDim.Width + w
							bottomDiffSlice[bottomIndex] += topDiffSlice[poolIndex] / float32(poolSize)
						}
					}
				}
			}
		}
	}
}

func (l *PoolingLayer) Params() []*Blob {
	return nil
}
