package main

import (
	"github.com/flammit/godnn"
	"log"
	"math/rand"
)

func TestRandomData(n int) godnn.DataLayer {
	data := [][][]float32{
		make([][]float32, n),
		make([][]float32, n),
	}
	for i := 0; i < n; i++ {
		d := float32(rand.Int() % 2)
		e := float32(rand.Int()%2) - 0.5
		f := float32(rand.Int()%2) - 0.5
		r := (rand.Float32() - 0.5) * 0.05
		data[0][i] = []float32{d + r + (0.05 * e), e + rand.Float32(), f + rand.Float32(), rand.Float32()}
		data[1][i] = []float32{d}
	}

	dataLayer := &godnn.FixedDataLayer{
		BaseLayer: godnn.BaseLayer{
			"random",
			nil,
			[]string{"data", "label"},
		},
		DataDims: []*godnn.BlobPoint{
			&godnn.BlobPoint{1, 1, 2, 2},
			&godnn.BlobPoint{1, 1, 1, 1},
		},
		Data: data,
	}
	return dataLayer
}

func main() {
	trainSize := 1000
	layers := []godnn.Layer{
		TestRandomData(trainSize),
		&godnn.FullyConnectedLayer{
			BaseLayer: godnn.BaseLayer{
				"ip",
				[]string{"data"},
				[]string{"ip"},
			},
			NumOutputs:  2,
			IncludeBias: false,
		},
		&godnn.SoftmaxWithLossLayer{
			BaseLayer: godnn.BaseLayer{
				"loss",
				[]string{"ip", "label"},
				[]string{"loss", "prob"},
			},
		},
	}
	net, err := godnn.NewNetwork(layers)
	if err != nil {
		log.Fatalln("failed to create network:", err)
	}
	log.Printf("Net Layers: %#v\n", net.Layers)
	for i, layer := range net.Layers {
		layerData := net.LayerDatas[i]
		log.Printf("Layer %d: %s\n", i, layer)
		for j, bottomBlob := range layerData.Bottom {
			log.Printf("Bottom Blob %d: %s\n", j, bottomBlob)
		}
		for j, topBlob := range layerData.Top {
			log.Printf("Top Blob %d: %s\n", j, topBlob)
		}
	}
	log.Printf("Net Params: %#v\n", net.Params)

	solver := godnn.NewSgdSolver(net)
	net.UpdateParams = true
	for i := 0; i < trainSize; i++ {
		loss := net.ForwardBackward()
		log.Printf("ForwardBackward %d: %f\n", i, loss)
		solver.ComputeUpdates()
		net.Update()
	}
}
