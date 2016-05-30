#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <climits>
#include <iostream>
#include <random>
#include <algorithm>
#include "parameters.h"
#include "hash.h"
#include "sketch.h"

using namespace std;

const char *filename_insert = "insert.dat";
const char *filename_query = "query.dat";

int main(int argc, char **argv)
{
    int ROWS_SKETCH = 4;
    int WIDTH_SKETCH = 30000;
    int factor = 10;
    if (argc >= 3) {
        ROWS_SKETCH = atoi(argv[1]);
        WIDTH_SKETCH = atoi(argv[2]);
    }
    if (argc >= 4)
        factor = atoi(argv[3]);
    if (argc >= 6) {
        N_INSERT = atoi(argv[4]);
        N_QUERY = atoi(argv[5]);
    }
    FILE *file_insert = fopen(filename_insert, "r");
    FILE *file_query = fopen(filename_query, "r");

    COUNT_SKETCH *cntsketch = new COUNT_SKETCH(ROWS_SKETCH ,WIDTH_SKETCH);
    CM_SKETCH *cmsketch = new CM_SKETCH(ROWS_SKETCH, WIDTH_SKETCH);
    FC1_SKETCH *fc1sketch = new FC1_SKETCH(ROWS_SKETCH, WIDTH_SKETCH);
    CML_SKETCH *cmlsketch = new CML_SKETCH(ROWS_SKETCH, WIDTH_SKETCH * 4, 1.08, 255);
    FID_SKETCH *fidsketch = new FID_SKETCH(ROWS_SKETCH, WIDTH_SKETCH, factor);
    H1_SKETCH *h1sketch = new H1_SKETCH(ROWS_SKETCH, WIDTH_SKETCH);
    BENCHMARK *benchmark = new BENCHMARK();
    vector<string> query_str;

    char str[LEN_STR + 1];
    int val;

    for (int i = 0; i < N_QUERY; ++i)
    {
        for (int j = 0; j < LEN_STR; ++j)
        {
            fscanf(file_query, "%d", &val);
            str[j] = val;
        }
        str[LEN_STR] = '\0';
        query_str.push_back(string(str));
    }

    //freopen("result.out", "w", stdout);
    for (int i = 0; i < N_INSERT; ++i)
    {
        for (int j = 0; j < LEN_STR; ++j)
        {
            fscanf(file_insert, "%d", &val);
            str[j] = val;
        }
        str[LEN_STR] = '\0';
        fscanf(file_insert, " : %d", &val);
        cntsketch->insert(str, val);
        cmsketch->insert(str, val);
        fc1sketch->insert(str, val);
        cmlsketch->insert(str, val);
        fidsketch->insert(str, val);
        h1sketch->insert(str, val);
        benchmark->insert(str, val);
        if ((i + 1) % (N_INSERT / 20))
            continue;
        double sum_cntsketch = 0;
        double sum_cmsketch = 0;
        double sum_fc1sketch = 0;
        double sum_cmlsketch = 0;
        double sum_fidsketch = 0;
        double sum_h1sketch = 0;
        double sum_benchmark = 0;
        int zero = 0;
        for (int j = 0; j < N_QUERY; ++j) {
            const char *str = query_str[j].c_str();
            int freq = benchmark->query(str);
            if (freq == 0) {
                ++zero;
                continue;
            }
            sum_cntsketch += fabs(cntsketch->query(str) - freq) / freq;
            sum_cmsketch += fabs(cmsketch->query(str) - freq) / freq;
            sum_fc1sketch += fabs(fc1sketch->query(str) - freq) / freq;
            sum_cmlsketch += fabs(cmlsketch->query(str) - freq) / freq;
            sum_fidsketch += fabs(fidsketch->query(str) - freq) / freq;
            sum_h1sketch += fabs(h1sketch->query(str) - freq) / freq;
            sum_benchmark += freq;
        }
        printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
        sum_cntsketch / (N_QUERY - zero),
        sum_cmsketch / (N_QUERY - zero),
        sum_fc1sketch / (N_QUERY - zero),
        sum_cmlsketch / (N_QUERY - zero),
        sum_fidsketch / (N_QUERY - zero),
        sum_h1sketch / (N_QUERY - zero),
        sum_benchmark);
    }

    fclose(file_insert);
    fclose(file_query);
    return 0;
}
