#ifndef HASH_H_INCLUDED
#define HASH_H_INCLUDED

inline unsigned hash1(const char *str)
{
    unsigned ret = 5381;
    for (int i = 0; i < LEN_STR; ++i)
        ret += (ret << 5) + (unsigned char)str[i];
    return ret;
}

inline unsigned hash2(const char *str)
{
    unsigned ret = 0;
    for (int i = 0; i < LEN_STR; ++i)
    {
        ret += (unsigned char)str[i];
        ret += (ret << 10);
        ret ^= (ret >> 6);
    }
    ret += (ret << 3);
    ret ^= (ret >> 11);
    ret += (ret << 15);
    return ret;
}

inline unsigned hash3(const char *str)
{
    unsigned ret = 0;
    for (int i = 0; i < LEN_STR; i++) {
        ret = ret * 33 + str[i];
    }
    return ret;
}

#endif // HASH_H_INCLUDED
