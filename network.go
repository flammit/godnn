package godnn

import (
	"errors"
	"log"
)

var (
	ErrUnreachableLayer = errors.New("invalid network definition, unreachable layers")
)

type Network struct {
	Layers       []Layer
	LayerDatas   []*LayerData
	UpdateParams bool

	BlobsByName map[string]*Blob
	Params      []*Blob
	LastParams  []*Blob
	ParamTemps  []*Blob
}

func NewNetwork(layers []Layer) (*Network, error) {
	n := new(Network)
	err := n.initLayers(layers)
	if err != nil {
		return nil, err
	}
	return n, nil
}

func (n *Network) initLayers(layers []Layer) error {
	// create layer data and connect layers via blob names
	n.Layers = make([]Layer, 0, len(layers))
	n.LayerDatas = make([]*LayerData, 0, len(layers))
	n.BlobsByName = make(map[string]*Blob)
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
	n.LastParams = make([]*Blob, len(n.Params))
	n.ParamTemps = make([]*Blob, len(n.Params))
	for i, param := range n.Params {
		n.LastParams[i] = NewBlob(param.Name+"_last", &param.Dim)
		n.ParamTemps[i] = NewBlob(param.Name+"_temp", &param.Dim)
	}
	return nil
}

func (n *Network) addLayer(layer Layer) error {
	layerData := new(LayerData)
	bottomNames := layer.BottomBlobNames()
	layerData.Bottom = make([]*Blob, len(bottomNames))
	for i, bottomName := range bottomNames {
		layerData.Bottom[i] = n.BlobsByName[bottomName]
	}
	err := layer.Setup(layerData)
	if err != nil {
		return err
	}
	for _, topBlob := range layerData.Top {
		n.BlobsByName[topBlob.Name] = topBlob
	}
	n.Layers = append(n.Layers, layer)
	n.LayerDatas = append(n.LayerDatas, layerData)
	n.Params = append(n.Params, layer.Params()...)
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
		layerData := n.LayerDatas[i]
		loss += layer.FeedForward(layerData)
	}
	return loss
}

func (n *Network) Backward(loss float32) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		layerData := n.LayerDatas[i]
		layer.FeedBackward(layerData, n.UpdateParams)
	}
}

func (n *Network) Update() {
	for _, param := range n.Params {
		paramDiff := param.Diff.CpuValues()
		paramData := param.Data.MutableCpuValues()
		Axpy32(len(paramDiff), +1, paramDiff, paramData)
	}
}
