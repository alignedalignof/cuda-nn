#ifndef CUDEF_H
#define CUDEF_H

#include <stdio.h>

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#define EXPECT(condition) do {\
		if (condition)\
			break;\
		fprintf(stderr, "False: %s @ %s: %u\n", #condition, __FILE__, __LINE__); \
		throw std::runtime_error("");\
	} while (0)

#endif
