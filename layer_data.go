package godnn

import (
	"errors"
)

var (
	ErrFixedLayerInvalidData = errors.New("invalid fixed layer data")
)

type DataLayer interface {
	Layer
	CurrentInputIndex() int
	NumInputs() int
}

type FixedDataLayer struct {
	BaseLayer
	DataDims   []*BlobPoint
	Data       [][][]float32
	numInputs  int
	inputIndex int
}

func (l *FixedDataLayer) Setup(d *LayerData) error {
	if len(l.Data) == 0 {
		return ErrFixedLayerInvalidData
	}
	if len(l.Data) != len(l.DataDims) {
		return ErrFixedLayerInvalidData
	}

	err := l.BaseLayer.checkNames(0, len(l.Data))
	if err != nil {
		return err
	}

	l.inputIndex = 0
	l.numInputs = len(l.Data[0])
	if l.numInputs == 0 {
		return ErrFixedLayerInvalidData
	}
	for i := 0; i < len(l.Data); i++ {
		data := l.Data[i]
		if len(data) != l.numInputs {
			return ErrFixedLayerInvalidData
		}
		dataDim := l.DataDims[i]
		dataSize := len(data[0])
		if dataDim.Size() != dataSize {
			return ErrFixedLayerInvalidData
		}
		for j := 0; j < len(data); j++ {
			if len(data[j]) != dataSize {
				return ErrFixedLayerInvalidData
			}
		}
	}

	d.Top = make([]*Blob, len(l.Data))
	for i := 0; i < len(l.Data); i++ {
		d.Top[i] = NewBlob(l.BaseLayer.TopNames[i], l.DataDims[i])
	}

	return nil
}

func (l *FixedDataLayer) FeedForward(d *LayerData) float32 {
	for i := 0; i < len(l.Data); i++ {
		Copy32(l.Data[i][l.inputIndex], d.Top[i].Data.MutableCpuValues(), l.DataDims[i].Size(), 0)
	}
	l.inputIndex++
	return 0
}

func (l *FixedDataLayer) FeedBackward(d *LayerData, paramPropagate bool) {}
func (l *FixedDataLayer) Params() []*Blob                                { return nil }
func (l *FixedDataLayer) CurrentInputIndex() int                         { return l.inputIndex }
func (l *FixedDataLayer) NumInputs() int                                 { return l.numInputs }

var _ = DataLayer(new(FixedDataLayer))
