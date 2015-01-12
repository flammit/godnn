package godnn

import (
	"encoding/binary"
	"errors"
	"github.com/boltdb/bolt"
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

var _ = DataLayer(new(FixedDataLayer))

func (l *FixedDataLayer) Setup(d *LayerData) error {
	if len(l.Data) == 0 {
		return ErrFixedLayerInvalidData
	}
	if len(l.Data) != len(l.DataDims) {
		return ErrFixedLayerInvalidData
	}

	err := l.checkNames(0, len(l.Data))
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
		d.Top[i] = NewBlob(l.TopNames[i], l.DataDims[i])
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

type BoltDbDataLayer struct {
	BaseLayer
	DbFileName string
	NumInBatch int
	db         *bolt.DB
	dims       []*BlobPoint
	numInputs  int
	inputIndex int
}

var _ = DataLayer(new(BoltDbDataLayer))

func intFromBytes(p []byte) int {
	return int(binary.LittleEndian.Uint32(p))
}

func intToBytes(n int) []byte {
	p := make([]byte, 4)
	binary.LittleEndian.PutUint32(p, uint32(n))
	return p
}

func (l *BoltDbDataLayer) getBucketDim(db *bolt.DB, bucketName string) (*BlobPoint, error) {
	dim := new(BlobPoint)
	db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName + "_dim"))
		dim.Batch = intFromBytes(b.Get([]byte("num")))
		dim.Channel = intFromBytes(b.Get([]byte("channel")))
		dim.Height = intFromBytes(b.Get([]byte("height")))
		dim.Width = intFromBytes(b.Get([]byte("width")))
		return nil
	})
	if dim.Batch == 0 || dim.Channel == 0 || dim.Height == 0 || dim.Width == 0 {
		return nil, errors.New("invalid bucket dimension for " + bucketName)
	}
	return dim, nil
}

func (l *BoltDbDataLayer) Setup(d *LayerData) (err error) {
	l.db, err = bolt.Open(l.DbFileName, 0400, nil)
	if err != nil {
		return err
	}

	l.dims = make([]*BlobPoint, len(l.TopNames))
	for i, bucketName := range l.TopNames {
		l.dims[i], err = l.getBucketDim(l.db, bucketName)
		if err != nil {
			return err
		}
		if i > 0 && l.dims[i].Batch != l.numInputs {
			return errors.New("each bucket needs the same number of inputs")
		}
		l.numInputs = l.dims[i].Batch
		l.dims[i].Batch = l.NumInBatch
	}

	d.Top = make([]*Blob, len(l.TopNames))
	for i, topName := range l.TopNames {
		d.Top[i] = NewBlob(topName, l.dims[i])
	}

	l.inputIndex = 0
	return nil
}

func (l *BoltDbDataLayer) FeedForward(d *LayerData) float32 {
	for n := 0; n < l.NumInBatch; n++ {
		inputKey := intToBytes(l.inputIndex)
		for topIndex, top := range d.Top {
			topData := top.Data.MutableCpuValues()
			topBucketName := l.TopNames[topIndex]

			topDataSlice := Subslice32(topData, n, l.dims[topIndex].BatchSize())
			l.db.View(func(tx *bolt.Tx) error {
				b := tx.Bucket([]byte(topBucketName))
				d := b.Get(inputKey)
				for i := 0; i < l.dims[topIndex].BatchSize(); i++ {
					topDataSlice[i] = float32(d[i]) / float32(256)
				}
				return nil
			})
		}
		l.inputIndex = (l.inputIndex + 1) % l.NumInputs()
	}
	return 0
}

func (l *BoltDbDataLayer) FeedBackward(d *LayerData, paramPropagate bool) {}
func (l *BoltDbDataLayer) Params() []*Blob                                { return nil }
func (l *BoltDbDataLayer) CurrentInputIndex() int                         { return l.inputIndex }
func (l *BoltDbDataLayer) NumInputs() int                                 { return l.numInputs }
