package godnn

type SoftmaxWithLossLayer struct {
	BaseLayer
	softmaxLayer     *SoftmaxLayer
	softmaxLayerData *LayerData
	prob             *Blob
	numBatches       int
	batchSize        int
	spatialSize      int
}

func (l *SoftmaxWithLossLayer) Setup(d *LayerData) error {
	err := l.BaseLayer.checkNames(2, 2)
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

var _ = Layer(new(SoftmaxWithLossLayer))
