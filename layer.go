package godnn

import (
	"errors"
	"log"
)

var (
	ErrInvalidBottomBlobNames = errors.New("invalid bottom blob names")
	ErrInvalidTopBlobNames    = errors.New("invalid top blob names")
)

type LayerData struct {
	Bottom []*Blob
	Top    []*Blob
	Params []*Blob
}

func (d *LayerData) DebugLayerData() {
	log.Printf("Layer Data: %#v\n", d)
	for j, bottomBlob := range d.Bottom {
		log.Printf("Bottom Blob %d: %s\n", j, bottomBlob)
	}
	for j, topBlob := range d.Top {
		log.Printf("Top Blob %d: %s\n", j, topBlob)
	}
}

type Layer interface {
	LayerName() string
	TopBlobNames() []string
	BottomBlobNames() []string

	Setup(d *LayerData) error
	FeedForward(d *LayerData) float32
	FeedBackward(d *LayerData, paramPropagate bool)
}

type BaseLayer struct {
	Name        string
	BottomNames []string
	TopNames    []string
}

func (l *BaseLayer) String() string            { return l.Name }
func (l *BaseLayer) LayerName() string         { return l.Name }
func (l *BaseLayer) TopBlobNames() []string    { return l.TopNames }
func (l *BaseLayer) BottomBlobNames() []string { return l.BottomNames }
func (l *BaseLayer) checkNames(expectedBottom, expectedTop int) error {
	err := l.checkBottomNames(expectedBottom)
	if err != nil {
		return nil
	}
	return l.checkTopNames(expectedTop)
}
func (l *BaseLayer) checkBottomNames(expected int) error {
	if len(l.BottomNames) != expected {
		return ErrInvalidBottomBlobNames
	}
	return nil
}
func (l *BaseLayer) checkTopNames(expected int) error {
	if len(l.TopNames) != expected {
		return ErrInvalidTopBlobNames
	}
	return nil
}
