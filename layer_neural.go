package godnn

type NeuronLayer struct {
	BaseLayer
	f ActivationFn
}

func (l *NeuronLayer) Setup(d *LayerData) error {
	err := l.BaseLayer.checkNames(1, 1)
	if err != nil {
		return err
	}

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.BaseLayer.TopNames[0], &d.Bottom[0].Dim)
	return nil
}

func (l *NeuronLayer) FeedForward(d *LayerData) float32 {
	bottomData := d.Bottom[0].Data.CpuValues()
	topData := d.Top[0].Data.MutableCpuValues()
	for i, v := range bottomData {
		topData[i] = l.f.Eval(v)
	}
	return 0
}

func (l *NeuronLayer) FeedBackward(d *LayerData, paramPropagate bool) {
	topDiff := d.Top[0].Diff.CpuValues()
	bottomData := d.Bottom[0].Data.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	for i, diff := range topDiff {
		bottomDiff[i] = diff * l.f.FirstDeriv(bottomData[i])
	}
}

func (l *NeuronLayer) Params() []*Blob { return nil }

func NewIdentityLayer(baseLayer BaseLayer) Layer {
	return &NeuronLayer{baseLayer, Identity}
}

func NewSigmoidLayer(baseLayer BaseLayer) Layer {
	return &NeuronLayer{baseLayer, Sigmoid}
}

func NewSoftsignLayer(baseLayer BaseLayer) Layer {
	return &NeuronLayer{baseLayer, Softsign}
}

func NewTanhLayer(baseLayer BaseLayer) Layer {
	return &NeuronLayer{baseLayer, Tanh}
}

var _ = Layer(new(NeuronLayer))
