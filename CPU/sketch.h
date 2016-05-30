#ifndef SKETCH_H_INCLUDED
#define SKETCH_H_INCLUDED

#include <queue>
#include <string>
#include <unordered_map>

using namespace std;

class BENCHMARK
{
private:
    unordered_map<string, int> counters;
public:
    void insert(char *str, int val)
    {
        str[LEN_STR] = '\0';
        string s(str);
        counters[s] = counters.count(s) == 0 ? val : counters[s] + val;
    }
    void remove(char *str, int val)
    {
        str[LEN_STR] = '\0';
        string s(str);
        int old = counters.count(s) == 0 ? 0 : counters[s];
        counters[s] = old > val ? old - val : 0;
    }
    int query(const char *str)
    {
        string s(str);
        return counters.count(s) == 0 ? 0 : counters[s];
    }
};

class SKETCH
{
protected:
    int d, w;
    int **counter;
public:
    SKETCH(int _d, int _w) : d(_d), w(_w - 3)
    {
        counter = new int* [d];
        for (int i = 0; i < d; ++i) {
            counter[i] = new int[_w];
            memset(counter[i], 0, sizeof(int) * _w);
        }
    }
    virtual void insert(const char *str, int val) = 0;
    virtual void remove(const char *str, int val) = 0;
    virtual int query(const char *str) = 0;
    virtual ~SKETCH()
    {
        for (int i = 0; i < d; ++i)
            delete [] counter[i];
        delete [] counter;
    }
};

class COUNT_SKETCH : public SKETCH
{
public:
    COUNT_SKETCH(int _d, int _w) : SKETCH(_d, _w) {}
    virtual void insert(const char *str, int val) {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        unsigned h3 = hash3(str);
        for (int i = 0; i < d; ++i) {
            int sign = (h3 & (1 << i)) ? 1 : -1;
            counter[i][(h1 + i * h2) % w] += val * sign;
        }
    }
    virtual void remove(const char *str, int val) {
        insert(str, -val);
    }
    int median(int *cnts) {
        sort(cnts, cnts + d);
        return (d & 1) ? cnts[d / 2] : (cnts[d / 2 - 1] + cnts[d / 2]) / 2;
    }
    virtual int query(const char *str) {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        unsigned h3 = hash3(str);
        int cnts[d];
        for (int i = 0; i < d; ++i) {
            int sign = (h3 & (1 << i)) ? 1 : -1;
            cnts[i] = counter[i][(h1 + i * h2) % w] * sign;
        }
        return median(cnts);
    }
    virtual ~COUNT_SKETCH() {}
};

class CML_SKETCH : public SKETCH
{
    int limit;
    double b;
    default_random_engine generator;
    uniform_real_distribution<double> distribution;
public:
    CML_SKETCH(int _d, int _w, double _b = 1.08, int _limit = 255) : b(_b), limit(_limit), SKETCH(_d, _w) {}
    bool decision(int c) {
        double r = distribution(generator);
        double lim = pow(b, -c);
        return r < lim;
    }
    double pointv(int c) {
        return c == 0 ? 0 : pow(b, c - 1);
    }
    virtual void insert(const char *str, int val)
    {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int k = 0; k < val; ++k) {
            int c = INT_MAX;
            for (int i = 0; i < d; ++i)
                if (counter[i][(h1 + i * h2) % w] < c)
                    c = counter[i][(h1 + i * h2) % w];
            if (c < limit && decision(c))
                for (int i = 0; i < d; ++i)
                    if (counter[i][(h1 + i * h2) % w] == c)
                        ++counter[i][(h1 + i * h2) % w];
        }
    }
    virtual void remove(const char *str, int val) {}
    virtual int query(const char *str) {
        int c = INT_MAX;
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int i = 0; i < d; ++i)
            if (counter[i][(h1 + i * h2) % w] < c)
                c = counter[i][(h1 + i * h2) % w];
        return c <= 1 ? pointv(c) : (int)(round((1 - pointv(c + 1)) / (1 - b)));
    }
    ~CML_SKETCH() {};
};

class CM_SKETCH : public SKETCH
{
public:
    CM_SKETCH(int _d, int _w) : SKETCH(_d, _w) {}
    virtual void insert(const char *str, int val)
    {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int i = 0; i < d; ++i)
            counter[i][(h1 + i * h2) % w] += val;
    }
    virtual void remove(const char *str, int val)
    {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w];
            *p = *p - val > 0 ? *p - val : 0;
        }
    }
    virtual int query(const char *str)
    {
        int ret = MAX_COUNTER;
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int i = 0; i < d; ++i)
        {
            int val = counter[i][(h1 + i * h2) % w];
            ret = ret < val ? ret : val;
        }
        return ret;
    }
    virtual ~CM_SKETCH() {}
};

class FC1_SKETCH : public CM_SKETCH
{
protected:
    int **cp_counter;
public:
    FC1_SKETCH(int _d, int _w) : CM_SKETCH(_d, _w) {
        cp_counter = new int* [_d];
        for (int i = 0; i < _d; ++i)
            cp_counter[i] = new int [_w];
    }
    virtual void insert(const char *str, int val)
    {
        int old = query(str);
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        int inc = 0;
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w];
            int *cp = &cp_counter[i][(h1 + i * h2) % w];
            int target = *p < old + val ? old + val : *p;
            *cp += *p + val - target;
            if (*p < old + val)
                ++inc;
            *p = target;
        }
    }
    virtual void remove(const char *str, int val)
    {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w],
                *cp = &cp_counter[i][(h1 + i * h2) % w];
            if (val > *cp) {
                *p -= val - *cp;
                *cp = 0;
            }
            else
                *cp -= val;
        }
    }
    virtual ~FC1_SKETCH() {
        for (int i = 0; i < d; ++i)
            delete [] counter[i];
        delete [] counter;
    }
};

class FC2_SKETCH : public FC1_SKETCH
{
protected:
    int insert_count;
    int pass_count;
    queue<pair<string, int>> waiting_list;
public:
    FC2_SKETCH(int _d, int _w)
        : FC1_SKETCH(_d, _w)
    {
        insert_count = 0;
        pass_count = 0;
    }
    virtual void insert(const char *str, int val)
    {
        ++insert_count;
        if (insert_count * 10 < N_INSERT)
        {
            FC1_SKETCH::insert(str, val);
            return;
        }
        ++pass_count;
        int old = query(str);
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        int inc = 0;
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w];
            if (*p < old + val)
                ++inc;
        }
        if (inc * 4 > d) {
            waiting_list.push(pair<string, int>(str, val));
            return;
        }
        FC1_SKETCH::insert(str, val);
    }
    void insert_all()
    {
        size_t threshhold = waiting_list.size() * 2;
        while (!waiting_list.empty() && --threshhold)
        {
            auto p = waiting_list.front();
            insert(p.first.c_str(), p.second);
            waiting_list.pop();
        }
        while (!waiting_list.empty())
        {
            auto p = waiting_list.front();
            FC1_SKETCH::insert(p.first.c_str(), p.second);
            waiting_list.pop();
        }
    }
};

class FID_SKETCH : public FC1_SKETCH {
protected:
    FC1_SKETCH fat_sketch;		// or we can use #CM_SKETCH fat_sketch;# to assist insertion
public:
    FID_SKETCH(int _d, int _w, int _k = 10)
    : FC1_SKETCH(_d, _w), fat_sketch(_d, _w * _k) {}
    virtual void insert(const char *str, int val) {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        int old = fat_sketch.query(str);
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w],
                *cp = &cp_counter[i][(h1 + i * h2) % w];
            int target = *p < old + val ? old + val : *p;
            *cp += *p + val - target;
            *p = target;
        }
        fat_sketch.insert(str, val);
    }
    virtual void remove(const char *str, int val) {
        FC1_SKETCH::remove(str, val);
        fat_sketch.remove(str, val);
    }
};

class H1_SKETCH : public FC1_SKETCH {
protected:
    BENCHMARK hashTable;
public:
    H1_SKETCH(int _d, int _w)
    : FC1_SKETCH(_d, _w), hashTable() {}
    virtual void insert(const char *str, int val) {
        unsigned h1 = hash1(str);
        unsigned h2 = hash2(str);
        int old = hashTable.query((char *)str);
        for (int i = 0; i < d; ++i) {
            int *p = &counter[i][(h1 + i * h2) % w],
                *cp = &cp_counter[i][(h1 + i * h2) % w];
            int target = *p < old + val ? old + val : *p;
            *cp += *p + val - target;
            *p = target;
        }
        hashTable.insert((char *)str, val);
    }
    virtual void remove(const char *str, int val) {
        FC1_SKETCH::remove(str, val);
        hashTable.remove((char *)str, val);
    }
};

#endif // SKETCH_H_INCLUDED
