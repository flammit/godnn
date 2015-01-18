package godnn

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/cgo"
	"math"
)

func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func Max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// TODO: use float32 ASM to optimize
func Pos32(x float32) float32 {
	if x >= 0 {
		return 1
	} else {
		return 0
	}
}
func Ceil32(x float32) float32  { return float32(math.Ceil(float64(x))) }
func Floor32(x float32) float32 { return float32(math.Floor(float64(x))) }
func Exp32(x float32) float32   { return float32(math.Exp(float64(x))) }
func Log32(x float32) float32   { return float32(math.Log(float64(x))) }
func Abs32(x float32) float32   { return float32(math.Abs(float64(x))) }
func Sq32(x float32) float32    { return x * x }
func Sqrt32(x float32) float32  { return float32(math.Sqrt(float64(x))) }
func Sign32(x float32) float32 {
	if math.Signbit(float64(x)) {
		return -1
	}
	return 1
}
func Add32(x, y float32) float32 { return x + y }
func Sub32(x, y float32) float32 { return x - y }
func Mul32(x, y float32) float32 { return x * y }
func Div32(x, y float32) float32 { return x / y }
func Max32(x, y float32) float32 { return float32(math.Max(float64(x), float64(y))) }
func Min32(x, y float32) float32 { return float32(math.Min(float64(x), float64(y))) }
func Pow32(x, n float32) float32 { return float32(math.Pow(float64(x), float64(n))) }

var (
	cpuBlas = new(cgo.Implementation)
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
	UnaryEval32(a, a, fn)
}

func UnaryEval32(a, y []float32, fn func(x float32) float32) {
	for i, x := range a {
		y[i] = fn(x)
	}
}

func BinaryApply32(a, b []float32, fn func(x, y float32) float32) {
	BinaryEval32(a, b, a, fn)
}

func BinaryEval32(a, b, y []float32, fn func(x, y float32) float32) {
	for i, x := range a {
		y[i] = fn(x, b[i])
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

func Im2col32(data_im []float32, channels, height, width, kernel_h, kernel_w,
	pad_h, pad_w, stride_h, stride_w int, data_col []float32) {
	height_col := (height+2*pad_h-kernel_h)/stride_h + 1
	width_col := (width+2*pad_w-kernel_w)/stride_w + 1
	channels_col := channels * kernel_h * kernel_w
	for c := 0; c < channels_col; c++ {
		w_offset := c % kernel_w
		h_offset := (c / kernel_w) % kernel_h
		c_im := c / kernel_h / kernel_w
		for h := 0; h < height_col; h++ {
			for w := 0; w < width_col; w++ {
				h_pad := h*stride_h - pad_h + h_offset
				w_pad := w*stride_w - pad_w + w_offset
				if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width {
					data_col[(c*height_col+h)*width_col+w] = data_im[(c_im*height+h_pad)*width+w_pad]
				} else {
					data_col[(c*height_col+h)*width_col+w] = 0
				}
			}
		}
	}
}

func Col2im32(data_col []float32, channels, height, width, patch_h, patch_w,
	pad_h, pad_w, stride_h, stride_w int, data_im []float32) {
	Set32(data_im, 0)
	height_col := (height+2*pad_h-patch_h)/stride_h + 1
	width_col := (width+2*pad_w-patch_w)/stride_w + 1
	channels_col := channels * patch_h * patch_w
	for c := 0; c < channels_col; c++ {
		w_offset := c % patch_w
		h_offset := (c / patch_w) % patch_h
		c_im := c / patch_h / patch_w
		for h := 0; h < height_col; h++ {
			for w := 0; w < width_col; w++ {
				h_pad := h*stride_h - pad_h + h_offset
				w_pad := w*stride_w - pad_w + w_offset
				if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width {
					data_im[(c_im*height+h_pad)*width+w_pad] += data_col[(c*height_col+h)*width_col+w]
				}
			}
		}
	}
}
