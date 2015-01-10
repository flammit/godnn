package godnn

import (
	"github.com/gonum/blas"
)

type InnerProductLayer struct {
	BaseLayer
	NumOutputs     int
	IncludeBias    bool
	m              int
	n              int
	k              int
	weightParams   *Blob
	biasParams     *Blob
	biasMultiplier *Blob
}

var _ = Layer(new(InnerProductLayer))

func (l *InnerProductLayer) Setup(d *LayerData) error {
	err := l.checkNames(1, 1)
	if err != nil {
		return err
	}

	bottomDim := d.Bottom[0].Dim
	l.m = bottomDim.Batch
	l.n = l.NumOutputs
	l.k = bottomDim.BatchSize()

	l.weightParams = NewBlob(l.LayerName()+"_weight", &BlobPoint{1, 1, l.n, l.k})
	if l.IncludeBias {
		l.biasParams = NewBlob(l.LayerName()+"_bias", &BlobPoint{1, 1, 1, l.n})
		l.biasMultiplier = NewBlob(l.LayerName()+"_biasMultiplier", &BlobPoint{1, 1, 1, l.m})
	}
	// TODO: initialize weights / bias / bias multiplier with filler

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.TopNames[0], &BlobPoint{l.m, l.n, 1, 1})

	return nil
}

func (l *InnerProductLayer) FeedForward(d *LayerData) float32 {
	bottomData := d.Bottom[0].Data.CpuValues()
	weightParams := l.weightParams.Data.CpuValues()
	topData := d.Top[0].Data.MutableCpuValues()
	Gemm32(blas.NoTrans, blas.Trans, l.m, l.n, l.k, 1, bottomData, weightParams, 0, topData)
	if l.IncludeBias {
		biasParams := l.biasParams.Data.CpuValues()
		biasMultiplier := l.biasMultiplier.Data.CpuValues()
		Gemm32(blas.NoTrans, blas.NoTrans, l.m, l.n, 1, 1, biasMultiplier, biasParams, 1, topData)
	}
	return 0
}

func (l *InnerProductLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	topDiff := d.Top[0].Diff.CpuValues()

	if paramPropagate {
		bottomData := d.Bottom[0].Data.CpuValues()
		weightParamDiffs := l.weightParams.Diff.MutableCpuValues()
		// Gradient w.r.t. weight
		Gemm32(blas.Trans, blas.NoTrans, l.n, l.k, l.m, 1, topDiff, bottomData, 0, weightParamDiffs)

		if l.IncludeBias {
			biasMultiplier := l.biasMultiplier.Data.CpuValues()
			biasParamsDiff := l.biasParams.Diff.MutableCpuValues()
			// Gradient w.r.t. bias
			Gemv32(blas.Trans, l.m, l.n, 1, topDiff, biasMultiplier, 0, biasParamsDiff)
		}
	}

	weightParams := l.weightParams.Data.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	// Gradient w.r.t. bottom data
	Gemm32(blas.NoTrans, blas.NoTrans, l.m, l.k, l.n, 1, topDiff, weightParams, 0, bottomDiff)
}

func (l *InnerProductLayer) Params() []*Blob {
	if l.IncludeBias {
		return []*Blob{l.weightParams, l.biasParams}
	}
	return []*Blob{l.weightParams}
}

func NewInnerProductLayer(baseLayer BaseLayer, numOutputs int, includeBias bool) *InnerProductLayer {
	return &InnerProductLayer{
		BaseLayer:   baseLayer,
		NumOutputs:  numOutputs,
		IncludeBias: includeBias,
	}
}

type SoftmaxLayer struct {
	BaseLayer
	sumMultiplier *Blob
	scale         *Blob
}

var _ = Layer(new(SoftmaxLayer))

func (l *SoftmaxLayer) Setup(d *LayerData) error {
	err := l.checkNames(1, 1)
	if err != nil {
		return err
	}

	bottomDim := d.Bottom[0].Dim
	l.sumMultiplier = NewBlob(l.LayerName()+"_sumMultiplier", &BlobPoint{1, bottomDim.Channel, 1, 1})
	Set32(l.sumMultiplier.Data.MutableCpuValues(), 1)
	l.scale = NewBlob(l.LayerName()+"_scale", &BlobPoint{1, 1, bottomDim.Height, bottomDim.Width})

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.TopNames[0], &d.Bottom[0].Dim)
	return nil
}

func (l *SoftmaxLayer) FeedForward(d *LayerData) float32 {
	bottomDim := d.Bottom[0].Dim
	batchSize := bottomDim.BatchSize()
	spatialSize := bottomDim.SpatialSize()

	bottomData := d.Bottom[0].Data.CpuValues()
	topData := d.Top[0].Data.MutableCpuValues()
	sumMultiplierData := l.sumMultiplier.Data.CpuValues()
	scaleData := l.scale.Data.MutableCpuValues()
	Copy32(bottomData, topData, bottomDim.Size(), 0)

	for batchIndex := 0; batchIndex < bottomDim.Batch; batchIndex++ {
		topSlice := Subslice32(topData, batchIndex, batchSize)

		Copy32(bottomData, scaleData, spatialSize, batchIndex*batchSize)
		for channelIndex := 0; channelIndex < bottomDim.Channel; channelIndex++ {
			for spatialIndex := 0; spatialIndex < spatialSize; spatialIndex++ {
				scaleData[spatialIndex] = Max32(scaleData[spatialIndex],
					bottomData[batchIndex*batchSize+channelIndex*spatialSize+spatialIndex])
			}
		}
		// Subtract the max
		Gemm32(blas.NoTrans, blas.NoTrans, bottomDim.Channel, spatialSize, 1,
			-1, sumMultiplierData, scaleData, 1, topSlice)

		// Exponentiate
		UnaryApply32(topSlice, Exp32)

		// Sum of Exponents
		Gemv32(blas.Trans, bottomDim.Channel, spatialSize,
			1, topSlice, sumMultiplierData, 0, scaleData)

		// Divide
		for channelIndex := 0; channelIndex < bottomDim.Channel; channelIndex++ {
			topSpatialSlice := Subslice32(topSlice, channelIndex, spatialSize)
			BinaryApply32(topSpatialSlice, scaleData, Div32)
		}
	}
	return 0
}

func (l *SoftmaxLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	bottomDim := d.Bottom[0].Dim
	batchSize := bottomDim.BatchSize()
	spatialSize := bottomDim.SpatialSize()

	topData := d.Top[0].Data.CpuValues()
	topDiff := d.Top[0].Diff.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	sumMultiplierData := l.sumMultiplier.Data.CpuValues()
	scaleData := l.scale.Data.MutableCpuValues()

	Copy32(topDiff, bottomDiff, bottomDim.Size(), 0)
	for batchIndex := 0; batchIndex < bottomDim.Batch; batchIndex++ {
		bottomDiffSlice := Subslice32(bottomDiff, batchIndex, batchSize)
		topDataSlice := Subslice32(topData, batchIndex, batchSize)
		for spatialIndex := 0; spatialIndex < spatialSize; spatialIndex++ {
			scaleData[spatialIndex] = Dot32(bottomDim.Channel,
				bottomDiffSlice[spatialIndex:], spatialSize,
				topDataSlice[spatialIndex:], spatialSize)
		}

		Gemm32(blas.NoTrans, blas.NoTrans, bottomDim.Channel, spatialSize, 1,
			-1, sumMultiplierData, scaleData, 1, bottomDiffSlice)
	}

	BinaryApply32(bottomDiff, topData, Mul32)
}

func (l *SoftmaxLayer) Params() []*Blob { return nil }

func NewSoftmaxLayer(baseLayer BaseLayer) *SoftmaxLayer {
	return &SoftmaxLayer{BaseLayer: baseLayer}
}
