#ifndef CUDA_NN
#define CUDA_NN

#include <memory>
#include <vector>

namespace CudaNn
	{

struct Cuda2dMem;

class Matrix
{
	std::shared_ptr<Cuda2dMem> mem_;

public:

	Matrix();
	Matrix(int rows, int cols);

	int rows() const;
	int columns() const;

	std::vector<float> get() const;
	void set(float* rows);

	void diag(float k);
	void rand(float k);
	Matrix max_row();
};

struct CudaNet;

class Net
{
	std::shared_ptr<CudaNet> net_;

public:
	Net(int in, const std::vector<int>& arch, int out);

	void fit(const Matrix X, const Matrix Y);
	Matrix evaluate(const Matrix X);
};

	}



#endif
