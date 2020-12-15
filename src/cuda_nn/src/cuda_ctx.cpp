#include <stdexcept>

#include <cuda.h>

using namespace std;

namespace CudaNn
	{

struct Cuda
{
	CUdevice dev;
	CUcontext ctx;
	Cuda();
	~Cuda();
};

Cuda::Cuda()
{
	dev = 0;
	ctx = 0;
	cuInit(0);
	if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
		cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
		cuCtxPushCurrent(ctx) != CUDA_SUCCESS)
	{
		if (ctx)
			cuDevicePrimaryCtxRelease(dev);
		throw runtime_error("No CUDA runtime");
	}

	char gpu[1000] = "Unknown";
	volatile CUresult cuda = cuDeviceGetName(gpu, sizeof(gpu), dev);
	printf("Using %s\n", gpu);
}

Cuda::~Cuda() {
	if (ctx)
		cuDevicePrimaryCtxRelease(dev);
}

extern "C" void cuda_ctx_init()
{
	static Cuda cuda;
}

	}
