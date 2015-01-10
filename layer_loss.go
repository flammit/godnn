package godnn

import (
	"errors"
)

type SoftmaxWithLossLayer struct {
	BaseLayer
	softmaxLayer     *SoftmaxLayer
	softmaxLayerData *LayerData
	prob             *Blob
	numBatches       int
	batchSize        int
	spatialSize      int
}

var _ = Layer(new(SoftmaxWithLossLayer))

func (l *SoftmaxWithLossLayer) Setup(d *LayerData) error {
	err := l.checkNames(2, 2)
	if err != nil {
		return err
	}

	l.softmaxLayerData = new(LayerData)
	l.softmaxLayerData.Bottom = []*Blob{d.Bottom[0]}
	l.softmaxLayer = &SoftmaxLayer{BaseLayer: BaseLayer{
		l.LayerName() + "_softmax",
		[]string{d.Bottom[0].Name},
		[]string{l.LayerName() + "_softmax_prob"},
	}}
	l.softmaxLayer.Setup(l.softmaxLayerData)

	l.prob = l.softmaxLayerData.Top[0]
	probDim := &l.prob.Dim
	l.numBatches = probDim.Batch
	l.batchSize = probDim.BatchSize()
	l.spatialSize = probDim.SpatialSize()

	d.Top = make([]*Blob, 2)
	d.Top[0] = NewBlob(l.TopNames[0], &BlobPoint{1, 1, 1, 1})
	d.Top[0].Diff.MutableCpuValues()[0] = 1 // loss weight
	d.Top[1] = NewBlob(l.TopNames[1], probDim)
	return nil
}

func (l *SoftmaxWithLossLayer) FeedForward(d *LayerData) float32 {
	l.softmaxLayer.FeedForward(l.softmaxLayerData)
	loss := float32(0)

	probData := l.prob.Data.CpuValues()
	labelData := d.Bottom[1].Data.CpuValues()
	for batchIndex := 0; batchIndex < l.numBatches; batchIndex++ {
		for spatialIndex := 0; spatialIndex < l.spatialSize; spatialIndex++ {
			loss -= Log32(Max32(probData[batchIndex*l.batchSize+
				int(labelData[batchIndex*l.spatialSize+spatialIndex])*l.spatialSize+spatialIndex],
				0))
		}
	}

	d.Top[0].Data.MutableCpuValues()[0] = loss / float32(l.numBatches*l.spatialSize)
	Copy32(probData, d.Top[1].Data.MutableCpuValues(), len(probData), 0)
	return loss
}

func (l *SoftmaxWithLossLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	probData := l.prob.Data.CpuValues()
	labelData := d.Bottom[1].Data.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	Copy32(probData, bottomDiff, len(probData), 0)
	for batchIndex := 0; batchIndex < l.numBatches; batchIndex++ {
		for spatialIndex := 0; spatialIndex < l.spatialSize; spatialIndex++ {
			index := batchIndex*l.batchSize +
				int(labelData[batchIndex*l.spatialSize+spatialIndex])*l.spatialSize + spatialIndex
			bottomDiff[index] -= 1
		}
	}
	lossWeight := d.Top[0].Diff.CpuValues()[0]
	Scal32(len(bottomDiff), lossWeight/float32(l.numBatches*l.spatialSize), bottomDiff)
}

func (l *SoftmaxWithLossLayer) Params() []*Blob { return nil }

var (
	ErrSigmoidCrossEntropyLossLayerInvalidInputSize = errors.New("invalid bottom data: must be the same size")
)

type SigmoidCrossEntropyLossLayer struct {
	BaseLayer
	sigmoidLayer     Layer
	sigmoidLayerData *LayerData
	bottomBatch      int
	bottomSize       int
}

var _ = Layer(new(SigmoidCrossEntropyLossLayer))

func (l *SigmoidCrossEntropyLossLayer) Setup(d *LayerData) error {
	err := l.checkNames(2, 1)
	if err != nil {
		return err
	}

	if d.Bottom[0].Dim != d.Bottom[1].Dim {
		return ErrSigmoidCrossEntropyLossLayerInvalidInputSize
	}

	// Setup Internal Sigmoid Layer
	l.sigmoidLayerData = new(LayerData)
	l.sigmoidLayerData.Bottom = []*Blob{d.Bottom[0]}
	l.sigmoidLayer = NewSigmoidLayer(BaseLayer{
		l.LayerName() + "_sigmoid",
		[]string{d.Bottom[0].Name},
		[]string{l.LayerName() + "_sigmoid_top"},
	})
	l.sigmoidLayer.Setup(l.sigmoidLayerData)

	bottomDim := d.Bottom[0].Dim
	l.bottomBatch = bottomDim.Batch
	l.bottomSize = bottomDim.Size()

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.TopNames[0], &BlobPoint{1, 1, 1, 1})
	return nil
}

func (l *SigmoidCrossEntropyLossLayer) FeedForward(d *LayerData) float32 {
	// Compute sigmoid outputs
	l.sigmoidLayer.FeedForward(l.sigmoidLayerData)

	// Compute the loss (negative log likelihood)
	loss := float32(0)
	inputData := d.Bottom[0].Data.CpuValues()
	targetData := d.Bottom[1].Data.CpuValues()
	for n := 0; n < l.bottomSize; n++ {
		loss -= (inputData[n] * (targetData[n] * Pos32(inputData[n]))) -
			Log32(1+Exp32(inputData[n]-(2*inputData[n]*Pos32(inputData[n]))))
	}

	d.Top[0].Data.MutableCpuValues()[0] = loss / float32(l.bottomBatch)
	return loss
}

func (l *SigmoidCrossEntropyLossLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	sigmoidOutputData := l.sigmoidLayerData.Top[0].Data.CpuValues()
	targetData := d.Bottom[1].Data.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	BinaryEval32(sigmoidOutputData, targetData, bottomDiff, Sub32)
	loss := d.Top[0].Data.CpuValues()[0]
	Scal32(l.bottomSize, loss/float32(l.bottomBatch), bottomDiff)
}

func (l *SigmoidCrossEntropyLossLayer) Params() []*Blob {
	return nil
}
