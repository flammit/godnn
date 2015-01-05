package godnn

import (
	"fmt"
)

type SyncedData struct {
	values []float32
	// TODO: add reference to be able to sync from GPU
	cpuDirty bool
	gpuDirty bool
}

func (d *SyncedData) Size() int {
	return len(d.values)
}

func (d *SyncedData) Sum() float32 {
	if d.gpuDirty {
		// TODO: implement with sum on GPU w/o data sync
		d.copyFromGpuToCpu()
	}
	values := d.CpuValues()
	return Asum32(len(values), values)
}

func (d *SyncedData) CpuValues() []float32 {
	if d.gpuDirty {
		d.copyFromGpuToCpu()
	}
	return d.values
}

func (d *SyncedData) GpuValues() {
	if d.cpuDirty {
		d.copyFromCpuToGpu()
	}
}

func (d *SyncedData) MutableCpuValues() []float32 {
	d.cpuDirty = true
	return d.values
}

func (d *SyncedData) MutableGpuValues() {
	d.gpuDirty = true
}

func (d *SyncedData) copyFromGpuToCpu() {
	// TODO: implement
	d.gpuDirty = false
}

func (d *SyncedData) copyFromCpuToGpu() {
	// TODO: implement
	d.cpuDirty = false
}

func NewSyncedData(capacity int) *SyncedData {
	return &SyncedData{make([]float32, capacity), false, false}
}

type BlobPoint struct {
	Batch   int
	Channel int
	Height  int
	Width   int
}

func (p BlobPoint) Size() int {
	return p.Batch * p.BatchSize()
}

func (p BlobPoint) BatchSize() int {
	return p.Channel * p.SpatialSize()
}

func (p BlobPoint) SpatialSize() int {
	return p.Height * p.Width
}

func (p BlobPoint) String() string {
	return fmt.Sprintf("(%d,%d,%d,%d)", p.Batch, p.Channel, p.Height, p.Width)
}

type Blob struct {
	Name string
	Dim  BlobPoint
	Data *SyncedData
	Diff *SyncedData
}

func (b *Blob) Offset(p *BlobPoint) int {
	return ((p.Batch*b.Dim.Channel+p.Channel)*b.Dim.Height+p.Height)*b.Dim.Width + p.Width
}

func (b *Blob) DataAt(p *BlobPoint) float32 {
	return b.Data.CpuValues()[b.Offset(p)]
}

func (b *Blob) DiffAt(p *BlobPoint) float32 {
	return b.Diff.CpuValues()[b.Offset(p)]
}

func (b *Blob) alloc(dim *BlobPoint) {
	b.Dim = *dim
	capacity := dim.Size()
	b.Data = NewSyncedData(capacity)
	b.Diff = NewSyncedData(capacity)
}

func (b *Blob) String() string {
	return fmt.Sprintf("%s: dim=%s", b.Name, b.Dim.String())
}

func NewBlob(name string, dim *BlobPoint) *Blob {
	b := new(Blob)
	b.Name = name
	b.alloc(dim)
	return b
}
