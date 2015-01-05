package godnn

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/cblas"
	"math"
)

// TODO: use ASM to optimize
func Floor32(x float32) float32 { return float32(math.Floor(float64(x))) }
func Exp32(x float32) float32   { return float32(math.Exp(float64(x))) }
func Log32(x float32) float32   { return float32(math.Log(float64(x))) }
func Abs32(x float32) float32   { return float32(math.Abs(float64(x))) }
func Sq32(x float32) float32    { return x * x }
func Sign32(x float32) float32 {
	if math.Signbit(float64(x)) {
		return -1
	}
	return 1
}
func Mul32(x, y float32) float32 { return x * y }
func Div32(x, y float32) float32 { return x / y }
func Max32(x, y float32) float32 { return float32(math.Max(float64(x), float64(y))) }
func Pow32(x, n float32) float32 { return float32(math.Pow(float64(x), float64(n))) }

var (
	cpuBlas = new(cblas.Blas)
)

func Subslice32(a []float32, offset, size int) []float32 {
	return a[offset*size : (offset+1)*size]
}

func Set32(a []float32, value float32) {
	for i, _ := range a {
		a[i] = value
	}
}

func Copy32(a, b []float32, count, offset int) {
	for i, j := offset, 0; i < offset+count; i++ {
		b[j] = a[i]
		j++
	}
}

func UnaryApply32(a []float32, fn func(x float32) float32) {
	for i, x := range a {
		a[i] = fn(x)
	}
}

func BinaryApply32(a, b []float32, fn func(x, y float32) float32) {
	for i, x := range a {
		a[i] = fn(x, b[i])
	}
}

func Gemm32(transA, transB blas.Transpose, m, n, k int,
	alpha float32, a, b []float32, beta float32, c []float32) {
	var lda, ldb int
	if transA == blas.NoTrans {
		lda = k
	} else {
		lda = m
	}
	if transB == blas.NoTrans {
		ldb = n
	} else {
		ldb = k
	}
	ldc := n
	cpuBlas.Sgemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func Gemv32(transA blas.Transpose, m, n int,
	alpha float32, a, x []float32, beta float32, y []float32) {
	lda := n
	incX := 1
	incY := 1
	cpuBlas.Sgemv(transA, m, n, alpha, a, lda, x, incX, beta, y, incY)
}

func Dot32(n int, x []float32, incX int, y []float32, incY int) float32 {
	return cpuBlas.Sdot(n, x, incX, y, incY)
}

func Axpy32(n int, alpha float32, x []float32, y []float32) {
	cpuBlas.Saxpy(n, alpha, x, 1, y, 1)
}

func Scal32(n int, alpha float32, x []float32) {
	cpuBlas.Sscal(n, alpha, x, 1)
}

func Asum32(n int, x []float32) float32 {
	return cpuBlas.Sasum(n, x, 1)
}
