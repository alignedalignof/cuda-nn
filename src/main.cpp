#include <stdexcept>
#include <vector>
#include <cmath>
#include <ctime>

#include <zlib.h>

#include "cuda_nn/cuda_nn.h"

using namespace std;
using namespace CudaNn;

const uint32_t MNIST_IMAGE = 2051;
const uint32_t MNIST_LABEL = 2049;
const int MNIST_LABELS = 10;
const int BATCH_SIZE = 1000;

static clock_t start = clock();
static void msprintf(const char* fmt, ...)
{
	long ms = 1000*(clock() - start);
	ms /= CLOCKS_PER_SEC ;
	printf("[%li ms] ", ms);
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	printf("\r\n");
}

static uint32_t big_endian_to_uint32(const uint8_t* bytes)
{
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
}

struct MnistFile
{
	uint32_t magic;
	uint32_t count;
	uint32_t height;
	uint32_t width;
	vector<uint8_t> data;
};

static MnistFile load_mnist(const char* file)
{
	struct Raii
	{
		gzFile tar;
		Raii(const char* file) { tar = gzopen(file, "rb"); }
		~Raii() { if (tar) gzclose(tar); }
	} raii(file);

	gzFile tar = raii.tar;
	if (!tar)
		throw runtime_error(string("Failed to open: ") + file);

	uint8_t hdr[8];
	if (gzread(tar, hdr, sizeof(hdr)) != sizeof(hdr))
		throw runtime_error("Failed reading header");

	MnistFile mnist;
	mnist.magic = big_endian_to_uint32(hdr);
	mnist.count = big_endian_to_uint32(hdr + 4);

	if (mnist.magic == MNIST_LABEL)
	{
		mnist.height = 0;
		mnist.width = 0;
		mnist.data.resize(mnist.count);
	}
	else if (mnist.magic == MNIST_IMAGE)
	{
		if (gzread(tar, hdr, sizeof(hdr)) != sizeof(hdr))
			throw runtime_error("Failed reading header");
		mnist.height = big_endian_to_uint32(hdr);
		mnist.width = big_endian_to_uint32(hdr + 4);
		mnist.data.resize(mnist.count*mnist.width*mnist.height);
	}
	else
	{
		throw runtime_error("Unknown MNIST file");
	}

	if (gzread(tar, mnist.data.data(), mnist.data.size()) != mnist.data.size())
		throw runtime_error(gzerror(tar, 0));
	return mnist;
}

struct Batch
{
	Matrix X;
	Matrix Y;
};

static vector<float> normalized_image_batch(MnistFile& images, int ofs, int count)
{
	if (images.magic != MNIST_IMAGE)
		throw runtime_error("No");

	int pixels = images.width*images.height;
	vector<float> batch(count*pixels);
	uint8_t* src = images.data.data() + ofs*pixels;
	for (int img = 0; img < count; img++)
	{
		float* dst = batch.data() + img;
		for (int pixel = 0; pixel < pixels; pixel++)
		{
			*dst = src[pixel]/255.;
			dst += count;
		}
		src += pixels;
	}
	return batch;
}

static vector<float> normalized_label_batch(MnistFile& labels, int ofs, int count)
{
	if (labels.magic != MNIST_LABEL)
		throw runtime_error("No");

	vector<float> batch(MNIST_LABELS*count);
	uint8_t* src = labels.data.data() + ofs;
	for (int label = 0; label < count; label++)
	{
		float* dst = batch.data() + label;
		for (int dim = 0; dim < MNIST_LABELS; dim++)
		{
			if (*src >= MNIST_LABELS)
				throw runtime_error("No");
			*dst = (*src == dim) ? 1.0 : 0.0;
			dst += count;
		}
		src++;
	}
	return batch;
}

static Batch create_batch(MnistFile& images, MnistFile& labels, int ofs, int count)
{
	if (images.magic != MNIST_IMAGE || labels.magic != MNIST_LABEL)
		throw runtime_error("No");
	if (images.count != labels.count)
		throw runtime_error("No");

	Batch batch;
	batch.X = Matrix(images.width*images.height, count);
	batch.Y = Matrix(10, count);

	batch.X.set(normalized_image_batch(images, ofs, count).data());
	batch.Y.set(normalized_label_batch(labels, ofs, count).data());

	return batch;
}

static float evaluate_prediction(Net& net, Batch& batch)
{
	auto H = net.evaluate(batch.X).max_row().get();
	auto Y = batch.Y.max_row().get();

	int M = H.size();
	int hit = 0;
	for (int i = 0; i < M; i++)
		hit += H[i] == Y[i];
	return hit/(float)M;
}

int main(int argn, char* argv[])
{
	string dir = (argn == 2) ? argv[1] : "../dat/fashion/";

	const int BATCH_COUNT = 6000;

	auto msload = [&dir](const char* file)
	{
		auto loc = dir + file;
		msprintf("Parsing %s", loc.c_str());
		return load_mnist(loc.c_str());
	};

	MnistFile images = msload("train-images-idx3-ubyte.gz");
	MnistFile labels = msload("train-labels-idx1-ubyte.gz");
	MnistFile test_images = msload("t10k-images-idx3-ubyte.gz");
	MnistFile test_labels = msload("t10k-labels-idx1-ubyte.gz");

	vector<Batch> batches;
	for (int ofs = 0; ofs < images.count; ofs += BATCH_COUNT)
	{
		msprintf("Uploading train images @%i[%i]", ofs, BATCH_COUNT);
		auto batch = create_batch(images, labels, ofs, BATCH_COUNT);
		batches.push_back(batch);
	}

	msprintf("Uploading test images");
	auto tests = create_batch(test_images, test_labels, 0, test_images.count);

	Net net(images.width*images.height, {(int)images.width, (int)images.width}, MNIST_LABELS);
	for (int epoch = 1; epoch <= 10000; epoch++)
	{
		for (auto& batch : batches)
			net.fit(batch.X, batch.Y);

		if ((epoch != 1) && (epoch % 500))
			continue;

		float a = 100*evaluate_prediction(net, batches[0]);
		float b = 100*evaluate_prediction(net, tests);
		msprintf("Epoch %i, train %.1lf%%, test %.1lf%%", epoch, a, b);
	}
	return 0;
}
