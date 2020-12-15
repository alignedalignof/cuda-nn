#include <unordered_map>
#include <stdexcept>
#include <type_traits>

#include <cuda.h>

#include "../cuda_nn.h"
#include "cudef.h"
#include "cuda_mem.h"
#include "kernelmat.h"

using namespace std;

	namespace CudaNn
	{

static_assert(is_standard_layout<Matrix>::value);

extern "C" void cuda_ctx_init();

static struct CudaNetCtx
{
	CUfunction reluFwd;
	CUfunction sigFwd;
	CUfunction sigBwdZ;
	CUfunction reluBwdZ;
	CUfunction reluBwdW;
	CUfunction reluBwdB;

	CudaNetCtx()
	{
		cuda_ctx_init();

		CUmodule module;
		EXPECT(cuModuleLoad(&module, "cudanet.ptx") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&reluFwd, module, "CudaNetReluFwd") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&sigFwd, module, "CudaNetSigmoidFwd") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&sigBwdZ, module, "CudaNetSigmoidBwdZ") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&reluBwdZ, module, "CudaNetReluBwdZ") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&reluBwdW, module, "CudaNetReluBwdW") == CUDA_SUCCESS);
		EXPECT(cuModuleGetFunction(&reluBwdB, module, "CudaNetReluBwdB") == CUDA_SUCCESS);
	}
} ctx;


enum Type
{
	RELU,
	SIGMOID,
};

struct Layer
{
	Matrix W;
	Matrix B;
	Type type;
};

struct Pass
{
	Matrix A;
	Matrix dZ;
};

static void* cumem(Matrix& m)
{
	auto ptr = ((shared_ptr<Cuda2dMem>*)&m)->get();
	return ptr ? &ptr->hdr : 0;
}

struct CudaNet
{
	vector<Layer> layers;
	unordered_map<int, vector<Pass>> passes;

	CudaNet(int in, const vector<int>& arch, int out)
	{
		vector<int> n_in(arch);
		n_in.insert(n_in.begin(), in);
		vector<int> n_out(arch);
		n_out.push_back(out);

		for (int layer = 0; layer < n_in.size(); ++layer)
		{
			auto W = Matrix(n_out[layer], n_in[layer]);
			auto B = Matrix(1, n_out[layer]);
			W.rand(1.0/W.columns());
			B.diag(0);
			layers.push_back(Layer{W, B, layer == arch.size() ? SIGMOID : RELU});
		}
	}

	void forward_sigmoid(Matrix Ain, Matrix W, Matrix B, Matrix Aout)
	{
		int tpbx = 16;
		int tpby = 16;
		int bpgx = ROUND_UP(Aout.columns(), tpbx)/tpbx;
		int bpgy = ROUND_UP(Aout.rows(), tpby)/tpby;

		void* args[] = { cumem(Ain), cumem(W), cumem(B), cumem(Aout), };
		EXPECT(cuLaunchKernel(ctx.sigFwd, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
	}

	void forward_relu(Matrix Ain, Matrix W, Matrix B, Matrix Aout)
	{
		int tpbx = 32;
		int tpby = 8;
		int bpgx = ROUND_UP(Aout.columns(), 128)/128;
		int bpgy = ROUND_UP(Aout.rows(), 128)/128;

		void* args[] = { cumem(Ain), cumem(W), cumem(B), cumem(Aout), };
		EXPECT(cuLaunchKernel(ctx.reluFwd, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
	}

	void backward_sigmoid(Matrix A, Matrix Y, Matrix dZ)
	{
		EXPECT(A.columns() == Y.columns());
		EXPECT(A.rows() == Y.rows());

		int tpbx = 16;
		int tpby = 16;
		int bpgx = ROUND_UP(A.columns(), tpbx)/tpbx;
		int bpgy = ROUND_UP(A.rows(), tpby)/tpby;

		void* args[] = { cumem(A), cumem(Y), cumem(dZ), };
		EXPECT(cuLaunchKernel(ctx.sigBwdZ, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) == CUDA_SUCCESS);
	}

	void backward_relu(Matrix dZin, Matrix A, Matrix W, Matrix B, Matrix dZout)
	{
		int tpbx = 16;
		int tpby = 16;

		if (cumem(dZout))
		{
			EXPECT(W.rows() == dZin.rows());
			EXPECT(dZin.columns() == dZout.columns());
			EXPECT(W.columns() == dZout.rows());
			EXPECT(A.columns() == dZout.columns());
			EXPECT(A.rows() == dZout.rows());

			int bpgx = ROUND_UP(dZout.columns(), tpbx)/tpbx;
			int bpgy = ROUND_UP(dZout.rows(), tpby)/tpby;
			void* args_dz[] = { cumem(dZin), cumem(W), cumem(A), cumem(dZout), };
			EXPECT(cuLaunchKernel(ctx.reluBwdZ, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args_dz, NULL) == CUDA_SUCCESS);
		}

		EXPECT(dZin.rows() == W.rows());
		EXPECT(dZin.columns() == A.columns());
		EXPECT(W.columns() == A.rows());

		tpbx = 32;
		tpby = 8;
		int bpgx = ROUND_UP(W.columns(), 128)/128;
		int bpgy = ROUND_UP(W.rows(), 128)/128;
		void* args_dw[] = { cumem(dZin), cumem(A), cumem(W), };
		EXPECT(cuLaunchKernel(ctx.reluBwdW, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args_dw, NULL) == CUDA_SUCCESS);

		EXPECT(dZin.rows() == B.columns());

		tpbx = 32;
		tpby = 1;
		bpgx = 1;
		bpgy = B.columns();
		void* args_db[] = { cumem(dZin), cumem(B), };
		EXPECT(cuLaunchKernel(ctx.reluBwdB, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args_db, NULL) == CUDA_SUCCESS);
	}

	Matrix evaluate(const Matrix X)
	{
		EXPECT(layers[0].W.columns() == X.rows());

		auto& pass = passes[X.columns()];
		if (!pass.size())
			for (auto& l : layers)
				pass.push_back(Pass{ Matrix(l.W.rows(), X.columns()), Matrix(l.W.rows(), X.columns())});

		Matrix in = X;
		for (int layer = 0; layer < layers.size(); layer++)
		{
			auto l = layers[layer];
			auto fwd = (l.type == RELU) ? &forward_relu : &forward_sigmoid;
			(this->*fwd)(in, l.W, l.B, pass[layer].A);
			in = pass[layer].A;
		}
		return pass.back().A;
	}

	void fit(Matrix X, Matrix Y)
	{
		EXPECT(X.columns() == Y.columns());
		EXPECT(layers.back().W.rows() == Y.rows());

		evaluate(X);

		auto& pass = passes[X.columns()];
		backward_sigmoid(pass.back().A, Y, pass.back().dZ);
		for (int l = layers.size(); l --> 1;)
		{
			auto layer = layers[l];
			auto dZ = pass[l].dZ;
			auto p = pass[l - 1];
			backward_relu(dZ, p.A, layer.W, layer.B, p.dZ);
		}
		backward_relu(pass[0].dZ, X, layers[0].W, layers[0].B, Matrix());
	}
};

Net::Net(int in, const vector<int>& arch, int out)
{
	net_ = make_shared<CudaNet>(in, arch, out);
}

void Net::fit(Matrix X, Matrix Y)
{
	net_->fit(X, Y);
}

Matrix Net::evaluate(const Matrix X)
{
	return net_->evaluate(X);
}

	}
