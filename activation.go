package godnn

type ActivationFn interface {
	Eval(x float32) float32
	FirstDeriv(y float32) float32
	SecondDeriv(y float32) float32
}

type sigmoid struct{}

func (f sigmoid) Eval(x float32) float32        { return 1.0 / (1.0 + Exp32(-x)) }
func (f sigmoid) FirstDeriv(y float32) float32  { return y * (1.0 - y) }
func (f sigmoid) SecondDeriv(y float32) float32 { return f.FirstDeriv(y) * (1 - (2 * y)) }

var Sigmoid = new(sigmoid)

type softsign struct{}

func (f softsign) Eval(x float32) float32        { return x / (1 + Abs32(x)) }
func (f softsign) FirstDeriv(y float32) float32  { return Sq32(1 - Abs32(y)) }
func (f softsign) SecondDeriv(y float32) float32 { return -2 * Sign32(y) * Pow32((1-Abs32(y)), 3) }

var Softsign = new(softsign)

type identity struct{}

func (f identity) Eval(x float32) float32        { return x }
func (f identity) FirstDeriv(y float32) float32  { return 1 }
func (f identity) SecondDeriv(y float32) float32 { return 0 }

var Identity = new(identity)

type tanh struct{}

func (f tanh) Eval(x float32) float32 {
	exp2x := Exp32(2 * x)
	return (exp2x - 1) / (exp2x + 1)
}
func (f tanh) FirstDeriv(y float32) float32  { return 1 - Sq32(f.Eval(y)) }
func (f tanh) SecondDeriv(y float32) float32 { return -2 * f.Eval(y) * f.FirstDeriv(y) }

var Tanh = new(tanh)
