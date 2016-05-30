#include "sketch.h"

enum {
	OP_INSERT,
	OP_QUERY
};

static unsigned *fc_sketch, *cp_sketch, *cm_sketch;
// denotes fewest counters incremented, complementary and count-min sketch, correspondingly
static int nstreams = 8;
static int nthreads = 512;
// nthreads should be the multiple of nstreams

void readFile(unsigned op, const char *filename, command_t *h_commands, int n);
// op: OP_INSERT | OP_QUERY ==> the file type to read in
// this function takes a filename and reads n lines of data into the address denoted by pointer h_commands
void cudaCall(unsigned op, command_t *h_commands, int n);
// a wrapper function for executing CUDA kernels
void writeFile(const char *filename, command_t *h_commands, int n);
// basiclly the same as readFile(), but write the data to file

int main(int argc, char **argv) {
	command_t *h_inserts, *h_queries;

	cudaMallocHost((void **)&h_inserts, ROUNDUP(N_INSERT, nthreads) * sizeof(command_t));
	cudaMallocHost((void **)&h_queries, ROUNDUP(N_QUERY, nthreads) * sizeof(command_t));
	assert(WIDTH_SKETCH % sizeof(unsigned) == 0);
	cudaMalloc((void **)&fc_sketch, ROWS_SKETCH * WIDTH_SKETCH * sizeof(uint8_t));
	cudaMalloc((void **)&cp_sketch, ROWS_SKETCH * WIDTH_SKETCH * sizeof(uint8_t));
	cudaMalloc((void **)&cm_sketch, ROWS_SKETCH * WIDTH_SKETCH_L * sizeof(uint8_t));
	cudaMemset(fc_sketch, 0, ROWS_SKETCH * WIDTH_SKETCH * sizeof(uint8_t));
	cudaMemset(cp_sketch, 0, ROWS_SKETCH * WIDTH_SKETCH * sizeof(uint8_t));
	cudaMemset(cm_sketch, 0, ROWS_SKETCH * WIDTH_SKETCH_L * sizeof(uint8_t));

	readFile(OP_INSERT, "insert.dat", h_inserts, N_INSERT);
	readFile(OP_QUERY, "query.dat", h_queries, N_QUERY);

	cudaCall(OP_INSERT, h_inserts, N_INSERT);
	cudaDeviceSynchronize();
	cudaCall(OP_QUERY, h_queries, N_QUERY);

	cudaFree(h_inserts);
	cudaFree(h_queries);
	cudaFree(fc_sketch);
	cudaFree(cp_sketch);
	cudaFree(cm_sketch);

	writeFile("sketch.out", h_queries, N_QUERY);
}

void readFile(unsigned op, const char *filename, command_t *h_commands, int n) {
	FILE *fin = fopen(filename, "r");
	for (int i = 0; i < n; ++i) {
		int val = 0;
		for (int j = 0; j < LEN_STR; ++j) {
			fscanf(fin, "%d", &val);
			h_commands[i].str[j] = (unsigned)val;
		}
		if (op == OP_INSERT)
			fscanf(fin, " : %d", &h_commands[i].val);
		else if (op == OP_QUERY)
			h_commands[i].val = MAX_COUNTER;
	}
	fclose(fin);
}

inline void cudaCall(unsigned op, command_t *h_commands, int n) {
	int threads = nthreads;
	int blocks = ROUNDUP(n, nstreams * threads) / (nstreams * threads);
	command_t *d_commands;
	cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

	cudaMalloc((void **)&d_commands, n * sizeof(command_t));
	for (int i = 0; i < nstreams; ++i)
		cudaStreamCreate(&streams[i]);
	for (int i = 0; i < nstreams; ++i) {
		cudaMemcpyAsync(d_commands + i * n / nstreams,
				h_commands + i * n / nstreams,
				n * sizeof(command_t) / nstreams,
				cudaMemcpyHostToDevice,
				streams[i]);
		// multi strream asynchrous data transfer
		if (op == OP_INSERT)
			insertKernel<<<blocks, threads, 0, streams[i]>>>
				(d_commands + i * n / nstreams, fc_sketch, cp_sketch, cm_sketch);
		// kernel execuation (corresponding to each stream)
		else if (op == OP_QUERY) {
			queryKernel<<<blocks, threads, 0, streams[i]>>>
				(d_commands + i * n / nstreams, fc_sketch);
			cudaMemcpyAsync(h_commands + i * n / nstreams,
					d_commands + i * n / nstreams,
					n * sizeof(command_t) / nstreams,
					cudaMemcpyDeviceToHost,
					streams[i]);
		}
	}
	for (int i = 0; i < nstreams; ++i)
		cudaStreamDestroy(streams[i]);
	cudaFree(d_commands);
}

void writeFile(const char *filename, command_t *h_commands, int n) {
	FILE *fout = fopen(filename, "w");
	for (int i = 0; i < n; ++i)
		fprintf(fout, "%d\n", h_commands[i].val);
	fclose(fout);
}
