package godnn

type NeuronLayer struct {
	BaseLayer
	f ActivationFn
}

var _ = Layer(new(NeuronLayer))

func (l *NeuronLayer) Setup(d *LayerData) error {
	err := l.checkNames(1, 1)
	if err != nil {
		return err
	}

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.TopNames[0], &d.Bottom[0].Dim)
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

type ReLULayer struct {
	BaseLayer
	NegativeSlope float32
}

var _ = Layer(new(ReLULayer))

func (l *ReLULayer) Setup(d *LayerData) error {
	err := l.checkNames(1, 1)
	if err != nil {
		return err
	}

	d.Top = make([]*Blob, 1)
	d.Top[0] = NewBlob(l.TopNames[0], &d.Bottom[0].Dim)
	return nil
}

func (l *ReLULayer) FeedForward(d *LayerData) float32 {
	bottomData := d.Bottom[0].Data.CpuValues()
	topData := d.Top[0].Data.MutableCpuValues()
	negativeSlope := l.NegativeSlope
	for i, v := range bottomData {
		topData[i] = Max32(v, 0) + (negativeSlope * Min32(v, 0))
	}
	return 0
}

func (l *ReLULayer) FeedBackward(d *LayerData, paramPropagate bool) {
	topDiff := d.Top[0].Diff.CpuValues()
	bottomData := d.Bottom[0].Data.CpuValues()
	bottomDiff := d.Bottom[0].Diff.MutableCpuValues()
	negativeSlope := l.NegativeSlope
	for i, diff := range topDiff {
		if bottomData[i] >= 0 {
			bottomDiff[i] = diff
		} else {
			bottomDiff[i] = negativeSlope
		}
	}
}

func (l *ReLULayer) Params() []*Blob { return nil }

func NewReLULayer(baseLayer BaseLayer, negativeSlope float32) *ReLULayer {
	return &ReLULayer{baseLayer, negativeSlope}
}
