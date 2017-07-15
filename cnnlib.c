/**
 * @file cnnlib.c
 * @brief Convolutionalニューラルネットワークで必要な関数をまとめたソースファイル
 * @author masan4444
 * @date 2017/07/15
 */

#include "cnnlib.h"
#include "nnlib.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void padding(int M, int N, const float * x, float * y) {
    int i;
    for (i = 0; i < N + 2; i ++) {
        y[i] = 0;
        y[i + (M + 1)(N + 2)] = 0;
    }
    for (i = 1; i <= M; i ++) {
        y[i*(N + 2)] = 0;
        y[i*(N + 3) + (N + 1)] = 0;
    }
}

void conv(int M, int N, int m, int n, const float * x, const float * F, const float * b, float * y) {

}
