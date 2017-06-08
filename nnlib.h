#ifndef _NNLIB_H_
#define _NNLIB_H_

void add(int, const float *, float *);
void scale(int, float, float *);
void init(int, float, float *);
void rand_init(int n, unsigned seed, float * o);

void print(int m, int n, const float * x);

void fc(int m, int n, const float * x, const float * A, const float * b, float * y);
void relu(int m, const float * x, float * y);
void softmax(int m, const float * x, float * y);

void softmaxwithloss_bwd(int m, const float * y, unsigned char t, float * dx);
void relu_bwd(int m, const float * x, const float * dy, float * dx);
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx);

void shuffle(int n, int * x, unsigned seed);
#endif // _NNLIB_H_
