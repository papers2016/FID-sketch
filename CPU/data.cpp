#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>
#include <string>
#include <vector>
#include "parameters.h"

using namespace std;

extern int N_INSERT;
extern int N_QUERY;
extern int LEN_STR;

int N_SEED = 100000;

vector<string> seeds;

int main(int argc, char **argv) {
	if (argc >= 2)
		N_INSERT = atoi(argv[1]);
	if (argc >= 3)
		N_QUERY = atoi(argv[2]);
	seeds.clear();
	srand(time(NULL));
	N_SEED = N_QUERY;
	for (int i = 0; i < N_SEED; ++i) {
		char s[LEN_STR + 1];
		for (int j = 0; j < LEN_STR; ++j)
			s[j] = rand() % 127 + 1;
		s[LEN_STR] = '\0';
		seeds.push_back(string(s));
	}
	assert(RAND_MAX > N_SEED);
	freopen("insert.dat", "w", stdout);
	for (int i = 0; i < N_INSERT; ++i) {
		string str = seeds[rand() % N_SEED];
		for (int j = 0; j < LEN_STR; ++j)
			printf("%d ", str[j]);
		printf(": 1\n");
	}
	freopen("query.dat", "w", stdout);
	for (int i = 0; i < N_SEED; ++i) {
		string str = seeds[i];
		for (int j = 0; j < LEN_STR; ++j)
			printf("%d ", str[j]);
		printf("\n");
	}
	return 0;
}
