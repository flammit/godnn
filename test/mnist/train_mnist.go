package main

import (
	"github.com/flammit/godnn"
	"log"
)

func main() {
	layers := []godnn.Layer{
		&godnn.BoltDbDataLayer{
			BaseLayer: godnn.BaseLayer{
				"data",
				[]string{},
				[]string{"images", "labels"},
			},
			DbFileName: "train.db",
			NumInBatch: 64,
		},
		&godnn.ConvolutionLayer{
			BaseLayer: godnn.BaseLayer{
				"conv1",
				[]string{"images"},
				[]string{"conv1"},
			},
			NumOutputs:   20,
			NumGroups:    1,
			KernelHeight: 5,
			KernelWidth:  5,
			PadHeight:    0,
			PadWidth:     0,
			StrideHeight: 1,
			StrideWidth:  1,
			IncludeBias:  true,
		},
		&godnn.PoolingLayer{
			BaseLayer: godnn.BaseLayer{
				"pool1",
				[]string{"conv1"},
				[]string{"pool1", "pool1_mask"},
			},
			Method:       godnn.PoolMethodMax,
			KernelHeight: 2,
			KernelWidth:  2,
			PadHeight:    0,
			PadWidth:     0,
			StrideHeight: 2,
			StrideWidth:  2,
		},
		&godnn.ConvolutionLayer{
			BaseLayer: godnn.BaseLayer{
				"conv2",
				[]string{"pool1"},
				[]string{"conv2"},
			},
			NumOutputs:   50,
			NumGroups:    1,
			KernelHeight: 5,
			KernelWidth:  5,
			PadHeight:    0,
			PadWidth:     0,
			StrideHeight: 1,
			StrideWidth:  1,
			IncludeBias:  true,
		},
		&godnn.PoolingLayer{
			BaseLayer: godnn.BaseLayer{
				"pool2",
				[]string{"conv2"},
				[]string{"pool2", "pool2_mask"},
			},
			Method:       godnn.PoolMethodMax,
			KernelHeight: 2,
			KernelWidth:  2,
			PadHeight:    0,
			PadWidth:     0,
			StrideHeight: 2,
			StrideWidth:  2,
		},
		&godnn.FullyConnectedLayer{
			BaseLayer: godnn.BaseLayer{
				"ip1",
				[]string{"pool2"},
				[]string{"ip1"},
			},
			NumOutputs:  500,
			IncludeBias: true,
		},
		&godnn.ReLULayer{
			BaseLayer: godnn.BaseLayer{
				"ip1_relu",
				[]string{"ip1"},
				[]string{"ip1_relu"},
			},
			NegativeSlope: float32(0),
		},
		&godnn.FullyConnectedLayer{
			BaseLayer: godnn.BaseLayer{
				"ip2",
				[]string{"ip1_relu"},
				[]string{"ip2"},
			},
			NumOutputs:  10,
			IncludeBias: true,
		},
		&godnn.SoftmaxWithLossLayer{
			BaseLayer: godnn.BaseLayer{
				"loss",
				[]string{"ip2", "labels"},
				[]string{"loss", "prob"},
			},
		},
	}
	net, err := godnn.NewNetwork(layers)
	if err != nil {
		log.Fatalln("failed to create network: ", err)
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
	for i := 0; i < 1000; i++ {
		loss := net.ForwardBackward()
		log.Printf("ForwardBackward %d: %f\n", i, loss)
		solver.ComputeUpdates()
		net.Update()
	}
}
