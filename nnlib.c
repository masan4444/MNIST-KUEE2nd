#include "nnlib.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void add(int n, const float * x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] += x[i];
    }
}
void scale(int n, float x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] *= x;
    }
}
void init(int n, float x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = x;
    }
}
void rand_init(int n, unsigned seed, float * o) {
    srand(seed);
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = (float)rand() / (RAND_MAX / 2) - 1;
    }
}

void print(int m, int n, const float * x) {
    int i;
    int j;
    for (i = 0; i < m; i ++) {
        for (j = 0; j < n; j ++) {
            printf("%f ", x[n * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}
void fc(int m, int n, const float * x, const float * A, const float * b, float * y) {
    init(m, 0, y);
    int i;
    int j;
    for (i = 0; i < m; i ++) {
        for (j = 0; j < n; j ++) {
            y[i] += (A[i*n + j] * x[j]);
        }
        y[i] += b[i]; 
    }
}
void relu(int m, const float * x, float * y) {
    int i;
    for (i = 0; i < m; i ++) {
        if (x[i] > 0) {
            y[i] = x[i];
        } else {
            y[i] = 0;
        }
    }
}
void softmax(int m, const float * x, float * y) {
    float max = x[0];
    int i;
    for (i = 1; i < m; i ++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    float exp_sum = 0;
    for (i = 0; i < m; i ++) {
        exp_sum += exp(x[i] - max);
    }
    for (i = 0; i < m; i ++) {
        y[i] = exp(x[i] - max)/exp_sum;
    }
}
void softmaxwithloss_bwd(int m, const float * y, unsigned char t, float * dx) {
    int i;
    for (i = 0; i < m; i ++) {
       if (i == t) {
          dx[i] = y[i] - 1;
       } else {
          dx[i] = y[i];
       }
    }
}
void relu_bwd(int m, const float * x, const float * dy, float * dx) {
    int i; 
    for (i = 0; i < m; i ++) {
        if (x[i] > 0) {
            dx[i] = dy[i];
        } else {
            dx[i] = 0;
        }
    }
}
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx) {
    init(m * n, 0, dA);
    init(m, 0, db);
    init(m, 0, dx);
    int i;
    int j;
    for (i = 0; i < m; i ++) {
        for (j = 0; j < n; j ++) {
            dx[j] += (A[j + n*i] * dy[i]);
            dA[n*i + j] = dy[i] * x[j];
        }
        db[i] = dy[i];
    }
}
void shuffle(int n, int * x, unsigned seed) {
    int i;
    int j;
    int s;
    srand(seed);
    for (i = 0; i < n; i ++) {
       j = rand() % n;
       s = x[i];
       x[i] = x[j];
       x[j] = s;
    }
}
