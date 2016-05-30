#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "sys/time.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "hash.h"

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

#define N_ALL 1100000
#define N_INSERT 100000
#define N_QUERY (N_ALL - N_INSERT)
#define LEN_STR 13
#define ROWS_SKETCH 16
#define WIDTH_SKETCH 10000
#define WIDTH_MOD 9997
#define WIDTH_SKETCH_L 100000
#define WIDTH_MOD_L 99997
#define MAX_COUNTER (unsigned)((uint8_t)(~0))

// Round up to the nearest multiple of v
#define ROUNDUP(n, v) ((n) - 1 + (v) - ((n) - 1) % (v))

typedef struct {
	// increase counter by val for insert type
	// count-min answer for query type
	unsigned val;
	unsigned char str[LEN_STR + 1];
} command_t;

__constant__ unsigned d_prime[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
};

__forceinline__ __device__ void
atomicInsert(unsigned *addr, int off, unsigned val)
{
        unsigned old = *addr, assumed, target;
        do {
                assumed = old;
                target = ((assumed >> off) & MAX_COUNTER) + val;
                target = target > MAX_COUNTER ? MAX_COUNTER : target;
                target = (assumed & (~(MAX_COUNTER << off))) | (target << off);
                old = atomicCAS(addr, assumed, target);
        } while (assumed != old);
}

__forceinline__ __device__ void
atomicDelete(unsigned *fc_addr, unsigned *cp_addr, int off, unsigned val) {
	unsigned old = *cp_addr, assumed, target, fc_val;
	do {
		assumed = old;
		target = ((assumed >> off) & MAX_COUNTER);
		fc_val = target < val ? val - target : 0;
		target = target < val ? 0 : target - val;
		target = (assumed & (~(MAX_COUNTER << off))) | (target << off);
		old = atomicCAS(cp_addr, assumed, target);
	} while (assumed != old);

	if (fc_val == 0) return;
	old = *fc_addr;
	do {
		assumed = old;
		target = ((assumed >> off) & MAX_COUNTER);
		target = target < fc_val ? 0 : target - fc_val;
		target = (assumed & (~(MAX_COUNTER << off))) | (target << off);
		old = atomicCAS(fc_addr, assumed, target);
	} while (assumed != old);
}

__global__ void
insertKernel(command_t *d_commands, unsigned *fc_sketch, unsigned *cp_sketch, unsigned *cm_sketch)
{
	int commandIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned val = d_commands[commandIdx].val;

	unsigned hash1 = 5381, hash2 = 0;
	unsigned char *str = d_commands[commandIdx].str;
	for (int i = 0; i < LEN_STR - 1; ++i) {
		hash1 += (hash1 << 5) + str[i];
		hash2 += str[i];
		hash2 += (hash2 << 10);
		hash2 ^= (hash2 >> 6);
	}
	hash2 += (hash2 << 3);
	hash2 ^= (hash2 >> 11);
	hash2 += (hash2 << 15);

	unsigned minimum = MAX_COUNTER;

	for (int row = 0; row < ROWS_SKETCH; ++row) {
		int col = (hash1 + hash2 * row) % WIDTH_MOD_L;
		int i = row * WIDTH_SKETCH_L + col;
		int off = (i & 3) * 8;
		atomicInsert(&cm_sketch[i >> 2], off, val);
		unsigned counter = cm_sketch[i >> 2];
		counter = (counter >> off) & MAX_COUNTER;
		minimum = minimum < counter ? minimum : counter;
	}

	for (int row = 0; row < ROWS_SKETCH; ++row) {
		int col = (hash1 + hash2 * row) % WIDTH_MOD;
		int i = row * WIDTH_SKETCH + col;
		int off = (i & 3) * 8;
		unsigned counter = fc_sketch[i >> 2];
		counter = (counter >> off) & MAX_COUNTER;
		minimum = minimum < counter + val ? minimum : counter + val;
	}

	for (int row = 0; row < ROWS_SKETCH; ++row) {
		int col = (hash1 + hash2 * row) % WIDTH_MOD;
		int i = row * WIDTH_SKETCH + col;
		int off = (i & 3) * 8;
		unsigned counter = fc_sketch[i >> 2];
		counter = (counter >> off) & MAX_COUNTER;
		unsigned fc_val = minimum > counter ? minimum - counter : 0;
		fc_val = fc_val < val ? fc_val : val;
		unsigned cp_val = val - fc_val;
		if (fc_val > 0)
			atomicInsert(&fc_sketch[i >> 2], off, fc_val);
		if (cp_val > 0)
			atomicInsert(&cp_sketch[i >> 2], off, cp_val);
	}
}

__global__ void
queryKernel(command_t *d_commands, unsigned *d_sketch)
{
	int commandIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned ret = MAX_COUNTER;

	unsigned hash1 = 5381, hash2 = 0;
	unsigned char *str = d_commands[commandIdx].str;
	for (int i = 0; i < LEN_STR - 1; ++i) {
		hash1 += (hash1 << 5) + str[i];
		hash2 += str[i];
		hash2 += (hash2 << 10);
		hash2 ^= (hash2 >> 6);
	}
	hash2 += (hash2 << 3);
	hash2 ^= (hash2 >> 11);
	hash2 += (hash2 << 15);

	for (int row = 0; row < ROWS_SKETCH; ++row) {
		int col = (hash1 + hash2 * row) % WIDTH_MOD;
		int i = row * WIDTH_SKETCH + col;
		int off = (i & 3) * 8;
		int val = d_sketch[i >> 2];
		val = (val >> off) & MAX_COUNTER;
		ret = ret < val ? ret : val;
	}
	d_commands[commandIdx].val = ret;
}

__global__ void
deleteKernel(command_t *d_commands, unsigned *fc_sketch, unsigned *cp_sketch)
{
	int commandIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned val = d_commands[commandIdx].val;

	unsigned hash1 = 5381, hash2 = 0;
	unsigned char *str = d_commands[commandIdx].str;
	for (int i = 0; i < LEN_STR - 1; ++i) {
		hash1 += (hash1 << 5) + str[i];
		hash2 += str[i];
		hash2 += (hash2 << 10);
		hash2 ^= (hash2 >> 6);
	}
	hash2 += (hash2 << 3);
	hash2 ^= (hash2 >> 11);
	hash2 += (hash2 << 15);

	for (int row = 0; row < ROWS_SKETCH; ++row) {
		int col = (hash1 + hash2 * row) % WIDTH_MOD;
		int i = row * WIDTH_SKETCH + col;
		int off = (i & 3) * 8;
		atomicDelete(&fc_sketch[i >> 2], &cp_sketch[i >> 2], off, val);
	}
}
