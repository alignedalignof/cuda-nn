#include "kernelmat.h"
#include "curand_kernel.h"

#define K 0.1f

extern "C" __global__ void Cuda2dMemRandNormal(KernelMat* A, float k)
{
	curandState_t state;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	curand_init(clock(), row, col, &state);
	if (row < A->rows && col < A->cols)
		A->data[row*A->stride + col] = k*curand_normal(&state);
}

extern "C" __global__ void Cuda2dMemDiag(KernelMat* A, float k)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (row < A->rows && col < A->cols)
		A->data[row*A->stride + col] = (row == col) ? k : 0;
}

extern "C" __global__ void Cuda2dMemMaxRow(const KernelMat* A, KernelMat* M)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (col >= A->cols)
		return;
			
	int k = A->rows;
	int stride = A->stride;
	int i = 0;
	float m = A->data[col];
	while (k --> 1)
	{
		float a = A->data[k*stride + col];
		if (a > m)
		{
			m = a;
			i = k;
		}
	}
	M->data[col] = i;
}

__device__ static inline float MatMulCell(const KernelMat* A, const KernelMat* B, int row, int col)
{	
	int n = A->cols;
	const float* a_data = A->data + row*A->stride;
	const float* b_data = B->data + col;
	int b_stride = B->stride;
	
	float acc = 0;
	for (int i = 0; i < n; i++)
		acc += a_data[i]*b_data[i*b_stride];
	return acc;
}

__device__ static inline void MatMadLayer(const KernelMat* A, const KernelMat* B, const KernelMat* C, float a[8][4][32], float4 b[8][32], float4* acc)
{
	int ax = threadIdx.x & 1;
	int ay = threadIdx.x >> 1;
	const float* a_data = A->data + (128*blockIdx.y + 16*threadIdx.y + ay)*A->stride + 4*ax;
	const float* b_data = B->data + threadIdx.y*B->stride + blockIdx.x*128 + 4*threadIdx.x;
	const float* b_end = B->data + B->rows*B->stride;
	const float* a_end = A->data + A->rows*A->stride;
	
	for (int i = 16; i --> 0;)
		acc[i] = float4{ 0, 0, 0, 0 };

	int N = A->cols;
	int b_stride = B->stride;
	while (N > 0)
	{	
		N -= 8;
		
		b[threadIdx.y][threadIdx.x] = (b_data < b_end) ? ((float4*)b_data)[0] : float4{0, 0, 0, 0};
		b_data += 8*b_stride;

		float4 a4 = (a_data < a_end) ? ((float4*)a_data)[0] : float4{ 0, 0, 0, 0 };
		int x = threadIdx.x;
		x = ((x & 1) << 4) | (x >> 1);
		a[threadIdx.y][0][x] = a4.x;
		a[threadIdx.y][1][x] = a4.y;
		a[threadIdx.y][2][x] = a4.z;
		a[threadIdx.y][3][x] = a4.w;
		a_data += 8;
		
		__syncthreads();
		
		#pragma unroll
		for (x = 0; x < 8; x++)
		{
			float4 bx = ((float4*)&b[x])[threadIdx.x];
			#pragma unroll
			for (int y = 0; y < 16; y += 4)
			{
				a4 = ((float4*)&a[threadIdx.y][x & 3][y + ((x & 4) << 2)])[0];

				acc[y].x += a4.x*bx.x;
				acc[y].y += a4.x*bx.y;
				acc[y].z += a4.x*bx.z;
				acc[y].w += a4.x*bx.w;
				
				acc[y + 1].x += a4.y*bx.x;
				acc[y + 1].y += a4.y*bx.y;
				acc[y + 1].z += a4.y*bx.z;
				acc[y + 1].w += a4.y*bx.w;
				
				acc[y + 2].x += a4.z*bx.x;
				acc[y + 2].y += a4.z*bx.y;
				acc[y + 2].z += a4.z*bx.z;
				acc[y + 2].w += a4.z*bx.w;
				
				acc[y + 3].x += a4.w*bx.x;
				acc[y + 3].y += a4.w*bx.y;
				acc[y + 3].z += a4.w*bx.z;
				acc[y + 3].w += a4.w*bx.w;
			}
		}
		__syncthreads();
	}
	
	int row = 128*blockIdx.y + 16*threadIdx.y + threadIdx.x;
	a[threadIdx.y][0][threadIdx.x] = ((row < C->rows) && (threadIdx.x < 16)) ? C->data[row] : 0;
	__syncthreads();
	for (int y = 0; y < 16; y++)
	{
		float c = a[threadIdx.y][0][y];
		acc[y].x += c;
		acc[y].y += c;
		acc[y].z += c;
		acc[y].w += c;
	}
}

extern "C" __global__ void CudaNetReluFwd(const KernelMat* in, const KernelMat* W, const KernelMat* B, KernelMat* out)
{
	__shared__ float4 b[8][32];
	__shared__ __align__(alignof(float4)) float a[8][4][32];
	
	float4 acc[16];
	MatMadLayer(W, in, B, a, b, acc);
	
	int stride = out->stride;
	if (128*blockIdx.x + 4*threadIdx.x >= stride)
		return;
	float* o = out->data + (128*blockIdx.y + 16*threadIdx.y)*stride + 128*blockIdx.x + 4*threadIdx.x;
	float* o_end = out->data + out->rows*stride;
	for (int y = 0; y < 16; y++)
	{
		if (o >= o_end)
			return;
		acc[y].x = (acc[y].x > 0) ? acc[y].x : 0;
		acc[y].y = (acc[y].y > 0) ? acc[y].y : 0;
		acc[y].z = (acc[y].z > 0) ? acc[y].z : 0;
		acc[y].w = (acc[y].w > 0) ? acc[y].w : 0;
		((float4*)o)[0] = acc[y];
		o += stride;
	}
}

extern "C" __global__ void CudaNetSigmoidFwd(const KernelMat* in, const KernelMat* W, const KernelMat* B, KernelMat* out)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (row >= out->rows || col >= out->cols)
		return;
		
	float cell = MatMulCell(W, in, row, col);
	cell += B->data[row];
	
	out->data[row*out->stride + col] = 1.0/(1.0 + expf(-cell));
}

extern "C" __global__ void CudaNetSigmoidBwdZ(const KernelMat* A, const KernelMat* Y, KernelMat* dZ)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (row < dZ->rows && col < dZ->cols)
		dZ->data[row*dZ->stride + col] = A->data[row*A->stride + col] - Y->data[row*Y->stride + col];
}

//dZout ~~ W.T*dZin
extern "C" __global__ void CudaNetReluBwdZ(const KernelMat* dZin, const KernelMat* W, KernelMat* A, KernelMat* dZout)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (row >= dZout->rows || col >= dZout->cols)
		return;
		
	int k = dZin->rows;
	const float* w_data = W->data + row;
	const float* dzin_data = dZin->data + col;
	int w_stride = W->stride;
	int dzin_stride = dZin->stride;
		
	float acc = 0;
	for (int i = 0; i < k; i++)
		acc += w_data[i*w_stride]*dzin_data[i*dzin_stride];
	
	dZout->data[row*dZout->stride + col] = (A->data[row*A->stride + col] > 0) ? acc : 0;
}

//B = B - k/m * sum(dZ, axis = 1)
extern "C" __global__ void CudaNetReluBwdB(const KernelMat* dZ, KernelMat* B)
{
	const int m = dZ->cols;
	int row = blockIdx.y;
	int col = threadIdx.x;

	if (row >= dZ->rows)
		return;
	
	const float* dz = dZ->data + row*dZ->stride;
	float acc = 0;
	for (; col < m;  col += blockDim.x)
		acc += dz[col];
	
	__shared__ float accs[32];
	accs[threadIdx.x] = acc;
	__syncwarp();
	
	if (threadIdx.x != 0)
		return;
	acc = 0;
	for (int i = 0; i < 32; i++)
		acc += accs[i];
	B->data[row] -= K*acc/m;
}

//W = W - (k/m)*dZ*A.T
//TODO optimize
extern "C" __global__ void CudaNetReluBwdW(const KernelMat* dZ, const KernelMat* A, KernelMat* W)
{
	int ax = threadIdx.x & 1;
	int ay = threadIdx.x >> 1;
	const float* dz = dZ->data + (128*blockIdx.y + 16*threadIdx.y + ay)*dZ->stride + 4*ax;
	const float* a = A->data + (128*blockIdx.x + 16*threadIdx.y + ay)*A->stride + 4*ax;
	const float* dz_end = dZ->data + dZ->rows*dZ->stride;
	const float* a_end = A->data + A->rows*A->stride;
	
	__shared__ __align__(alignof(float4)) float dz_blk[8][128];
	__shared__ __align__(alignof(float4)) float a_blk[8][128];
	
	float4 acc[16];
	for (int i = 16; i --> 0;)
		acc[i] = float4{ 0, 0, 0, 0 };

	int N = dZ->cols;
	int layer = 4*ax;
	int bank = (16*(threadIdx.y + ax) + ay) & 0x7f;
	while (N > 0)
	{	
		float4 a4 = (a < a_end) ? ((float4*)a)[0] : float4{ 0, 0, 0, 0 };
		a_blk[layer][bank] = a4.x;
		a_blk[layer + 1][bank] = a4.y;
		a_blk[layer + 2][bank] = a4.z;
		a_blk[layer + 3][bank] = a4.w;
		a += 8;

		float4 dz4 = (dz < dz_end) ? ((float4*)dz)[0] : float4{ 0, 0, 0, 0 };
		dz_blk[layer][bank] = dz4.x;
		dz_blk[layer + 1][bank] = dz4.y;
		dz_blk[layer + 2][bank] = dz4.z;
		dz_blk[layer + 3][bank] = dz4.w;
		dz += 8;
		
		__syncthreads();
		
		#pragma unroll
		for (int l = 0; l < 8; l++)
		{
			int ofs = 4*(l & 4);
			a4 = ((float4*)&a_blk[l])[(threadIdx.x + ofs/4) & 0x1f];
			#pragma unroll
			for (int y = 0; y < 16; y += 4)
			{
				float4 dz4 = ((float4*)&dz_blk[l][(16*threadIdx.y + ofs + y) & 0x7f])[0];
				#pragma unroll
				for (int r = 0; r < 4; r++)
				{
					acc[y + r].x += ((float*)&dz4)[r]*a4.x;
					acc[y + r].y += ((float*)&dz4)[r]*a4.y;
					acc[y + r].z += ((float*)&dz4)[r]*a4.z;
					acc[y + r].w += ((float*)&dz4)[r]*a4.w;
				}
			}
			N--;
			if (N == 0)
				break;
		}
		__syncthreads();
	}
	
	int w_stride = W->stride;
	if (128*blockIdx.x + 4*threadIdx.x >= w_stride)
		return;
		
	float* w = W->data + (128*blockIdx.y + 16*threadIdx.y)*w_stride + 128*blockIdx.x + 4*threadIdx.x;
	float* w_end = W->data + W->rows*w_stride;
	int M = dZ->cols;
	for (int y = 0; y < 16; y++)
	{
		if (w >= w_end)
			return;
		float4 w4 = ((float4*)w)[0];
		w4.x -= K*acc[y].x/M;
		w4.y -= K*acc[y].y/M;
		w4.z -= K*acc[y].z/M;
		w4.w -= K*acc[y].w/M;
		((float4*)w)[0] = w4;
		w += w_stride;
	}
}