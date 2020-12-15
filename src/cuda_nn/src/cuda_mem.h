#ifndef CUDA_MAT
#define CUDA_MAT

#include <cuda.h>

struct CudaNn::Cuda2dMem
{
	CUdeviceptr hdr;
	CUdeviceptr data;

	int rows;
	int cols;
	int stride;
	size_t bytes;


	Cuda2dMem(int rows, int cols);
	~Cuda2dMem();

	void diag(float k);
	void upload(float* host);
	void download(float* host);
};

#endif
