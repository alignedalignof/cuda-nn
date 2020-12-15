#include <unordered_map>
#include <stdexcept>

#include <cuda.h>

#include "../cuda_nn.h"

#include "cuda_mem.h"
#include "cudef.h"
#include "kernelmat.h"

using namespace std;

namespace CudaNn
	{

extern "C" void cuda_ctx_init();

static struct Cuda2dMemCtx
{
	CUfunction diag;
	CUfunction maxRow;
	CUfunction rand;

	Cuda2dMemCtx()
	{
		cuda_ctx_init();

		CUmodule module;
		EXPECT(cuModuleLoad(&module, "cudanet.ptx") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&diag, module, "Cuda2dMemDiag") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&maxRow, module, "Cuda2dMemMaxRow") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&rand, module, "Cuda2dMemRandNormal") == CUDA_SUCCESS);
	}
} ctx2;

Cuda2dMem::Cuda2dMem(int rows, int cols)
{
	if (rows < 1 || cols < 1)
		throw invalid_argument("Invalid matrix size");
	this->rows = rows;
	this->cols = cols;

	const int PAD = alignof(KernelMat::data)/sizeof(float);
	stride = ROUND_UP(cols, PAD);
	bytes = stride*rows*sizeof(float);

	hdr = 0;
	KernelMat mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.stride = stride;
	if (cuMemAlloc(&hdr, sizeof(KernelMat) + bytes) != CUDA_SUCCESS ||
		cuMemcpyHtoD(hdr, &mat, sizeof(KernelMat)) != CUDA_SUCCESS) {
		if (hdr)
			cuMemFree(hdr);
		throw runtime_error("No GPU memory");
	}
	data = hdr + sizeof(KernelMat);
}

Cuda2dMem::~Cuda2dMem()
{
	cuMemFree(hdr);
}

void Cuda2dMem::diag(float k)
{
	int tpbx = 16;
	int tpby = 16;
	int bpgx = ROUND_UP(cols, tpbx)/tpbx;
	int bpgy = ROUND_UP(rows, tpby)/tpby;

	void* args[] = { &hdr, &k };
	EXPECT(cuLaunchKernel(ctx2.diag, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
}

void Cuda2dMem::upload(float* host)
{
	for (int row = 0; row < rows; ++row)
	{
		float* src = host + row*cols;
		CUdeviceptr dst = data + row*stride*sizeof(float);
		EXPECT(cuMemcpyHtoD(dst, src, cols*sizeof(float)) == CUDA_SUCCESS);
	}
}

void Cuda2dMem::download(float* host)
{
	for (int row = 0; row < rows; ++row)
	{
		CUdeviceptr src = data + row*stride*sizeof(float);
		float* dst = host + row*cols;
		EXPECT(cuMemcpyDtoH(dst, src, cols*sizeof(float)) == CUDA_SUCCESS);
	}
}

Matrix::Matrix()
{

}

Matrix::Matrix(int rows, int cols)
{
	mem_ = make_shared<Cuda2dMem>(rows, cols);
}

int Matrix::rows() const
{
	return mem_->rows;
}

int Matrix::columns() const
{
	return mem_->cols;
}

void Matrix::set(float* rows)
{
	mem_->upload(rows);
}

vector<float> Matrix::get() const
{
	vector<float> v(columns()*rows());
	mem_->download(v.data());
	return v;
}

void Matrix::diag(float k)
{
	mem_->diag(k);
}

void Matrix::rand(float k)
{
	int tpbx = 16;
	int tpby = 16;
	int bpgx = ROUND_UP(mem_->cols, tpbx)/tpbx;
	int bpgy = ROUND_UP(mem_->rows, tpby)/tpby;

	void* args[] = { &mem_->hdr, &k };
	EXPECT(cuLaunchKernel(ctx2.rand, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
}

Matrix Matrix::max_row()
{
	int tpbx = 32;
	int tpby = 1;
	int bpgx = ROUND_UP(mem_->cols, tpbx)/tpbx;
	int bpgy = 1;
	Matrix m(1, mem_->cols);

	void* args[] = { &mem_->hdr, &m.mem_->hdr };
	EXPECT(cuLaunchKernel(ctx2.maxRow, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
	return m;
}

	}
