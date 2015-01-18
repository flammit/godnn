package godnn

import (
	"errors"
	"log"
)

var (
	ErrUnreachableLayer = errors.New("invalid network definition, unreachable layers")
)

type Network struct {
	Layers          []Layer
	LayerDataByName map[string]*LayerData
	BlobsByName     map[string]*Blob
	UpdateParams    bool
}

func NewNetwork(layers []Layer) (*Network, error) {
	return NewNetworkFromTraining(layers, nil)
}

func NewNetworkFromTraining(layers []Layer, trainNet *Network) (*Network, error) {
	n := new(Network)
	n.Layers = make([]Layer, 0, len(layers))
	if trainNet != nil {
		n.LayerDataByName = trainNet.LayerDataByName
		n.BlobsByName = trainNet.BlobsByName
	} else {
		n.LayerDataByName = make(map[string]*LayerData, len(layers))
		n.BlobsByName = make(map[string]*Blob)
	}
	err := n.initLayers(layers)
	if err != nil {
		return nil, err
	}
	return n, nil
}

func (n *Network) initLayers(layers []Layer) error {
	// create layer data and connect layers via blob names
	added := make([]bool, len(layers))
finalLayer:
	for len(n.Layers) < len(layers) {
		// find a layer that can be added (bottom blobs defined) and push to final layer
		for i := 0; i < len(layers); i++ {
			layer := layers[i]
			if !added[i] && n.addableLayer(layer) {
				added[i] = true
				err := n.addLayer(layer)
				if err != nil {
					return err
				}
				continue finalLayer
			}
		}
		log.Println("Added Layers:", added)
		return ErrUnreachableLayer
	}
	return nil
}

func (n *Network) addLayer(layer Layer) error {
	var layerData *LayerData
	if data, ok := n.LayerDataByName[layer.LayerName()]; ok {
		layerData = data
	} else {
		layerData = new(LayerData)
		bottomNames := layer.BottomBlobNames()
		layerData.Bottom = make([]*Blob, len(bottomNames))
		for i, bottomName := range bottomNames {
			layerData.Bottom[i] = n.BlobsByName[bottomName]
		}
	}

	err := layer.Setup(layerData)
	if err != nil {
		return err
	}

	for _, topBlob := range layerData.Top {
		n.BlobsByName[topBlob.Name] = topBlob
	}

	n.Layers = append(n.Layers, layer)
	n.LayerDataByName[layer.LayerName()] = layerData
	return nil
}

func (n *Network) addableLayer(layer Layer) bool {
	for _, bottomName := range layer.BottomBlobNames() {
		_, ok := n.BlobsByName[bottomName]
		if !ok {
			return false
		}
	}
	return true
}

func (n *Network) ForwardBackward() float32 {
	loss := n.Forward()
	n.Backward(loss)
	return loss
}

func (n *Network) Forward() float32 {
	loss := float32(0)
	for i := 0; i < len(n.Layers); i++ {
		layer := n.Layers[i]
		layerData := n.LayerData(layer)
		loss += layer.FeedForward(layerData)
	}
	return loss
}

func (n *Network) Backward(loss float32) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		layerData := n.LayerData(layer)
		layer.FeedBackward(layerData, n.UpdateParams)
	}
}

func (n *Network) Update() {
	for _, layerData := range n.LayerDataByName {
		for _, param := range layerData.Params {
			paramDiff := param.Diff.CpuValues()
			paramData := param.Data.MutableCpuValues()
			Axpy32(len(paramDiff), +1, paramDiff, paramData)
		}
	}
}

func (n *Network) LayerData(layer Layer) *LayerData {
	return n.LayerDataByName[layer.LayerName()]
}

func (n *Network) Params() []*Blob {
	params := []*Blob{}
	for _, layerData := range n.LayerDataByName {
		params = append(params, layerData.Params...)
	}
	return params
}
