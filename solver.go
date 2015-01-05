package godnn

type Solver interface {
	ComputeUpdates()
}

type SgdSolver struct {
	Momentum         float32
	BaseLearningRate float32
	WeightDecay      float32
	Gamma            float32
	StepSize         int
	net              *Network

	lastParams []*Blob // diffs are last param diffs, data is temporary
	iterations int
}

func (s *SgdSolver) ComputeUpdates() {
	s.iterations++
	for i, param := range s.net.Params {
		paramData := param.Data.CpuValues()
		paramDiff := param.Diff.MutableCpuValues()

		lastParam := s.lastParams[i]
		paramTemp := lastParam.Data.MutableCpuValues()
		lastParamDiff := lastParam.Diff.MutableCpuValues()

		// Apply Weight Decay
		Axpy32(len(paramData), s.WeightDecay, paramData, paramDiff)

		// Compute Param Updates
		rate := s.calculateRate()
		Set32(paramTemp, 0)
		Axpy32(len(paramTemp), s.Momentum, lastParamDiff, paramTemp)
		Axpy32(len(paramTemp), -rate, paramDiff, paramTemp)
		Copy32(paramTemp, lastParamDiff, len(paramTemp), 0)
		Copy32(paramTemp, paramDiff, len(paramTemp), 0)
	}
}

func (s *SgdSolver) calculateRate() float32 {
	return s.BaseLearningRate * Pow32(s.Gamma, float32(s.iterations)/float32(s.StepSize))
}

func NewSgdSolver(net *Network) *SgdSolver {
	s := &SgdSolver{net: net}
	s.Momentum = float32(0.9)
	s.BaseLearningRate = float32(0.01)
	s.Gamma = float32(0.1)
	s.StepSize = 100000
	s.WeightDecay = float32(0.0005)

	s.lastParams = make([]*Blob, len(net.Params))
	for i, param := range net.Params {
		s.lastParams[i] = NewBlob(param.Name+"_solver_last", &param.Dim)
	}
	return s
}
