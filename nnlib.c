#include "nnlib.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @fn
 * 配列同士の足し算
 * @param (int n) 配列の個数
 * @param (const float * x) 足す配列
 * @param (float * o) 足される配列
 * @return 無し
 */
void add(int n, const float * x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] += x[i];
    }
}

/**
 * @fn
 * 配列同士の引き算
 * @param (int n) 配列の個数
 * @param (const float * x) 引く配列
 * @param (const float * y) 引かれる配列
 * @param (const flaot * o) 答(差)の配列
 * @return 無し
 * @detail MomentumSGDで使用
 */
void sub(int n, const float * x, const float * y, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = x[i] - y[i];
    }
}

/**
 * @fn
 * 配列を定数倍する
 * @param (int n) サイズ
 * @param (float x) かける数
 * @param (float * o) かけられる配列
 * @return 無し
 */
void scale(int n, float x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] *= x;
    }
}

/**
 * @fn
 * 配列yを定数倍し，配列oに足す
 * @param (int n) 配列のサイズ
 * @param (float x) 定数倍する数
 * @param (float * y) 定数倍され，足す配列
 * @param (float * o) 足される配列
 * @return 無し
 * @detail scaleとaddを別々に使うのが面倒であり，ループも無駄に倍回すので，一つにした
 */
void scale_and_add(int n, float x, const float * y, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] += y[i] * x;
    }
}

/**
 * @fn
 * 配列を定数で初期化する
 * @param (int n) 配列のサイズ
 * @param (float x) 定数
 * @param (float * o) 初期化される配列
 * @return 無し
 */
void init(int n, float x, float * o) {
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = x;
    }
}

/**
 * @fn
 * -1から1で配列を初期化する
 * @param (int n) 配列のサイズ
 * @param (unsigned seed) rand()のシード値
 * @param (float * o) 初期化される配列
 * @return 無し
 */
void rand_init(int n, unsigned seed, float * o) {
    srand(seed);
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = (float)rand() / (RAND_MAX / 2) - 1;
    }
}

/**
 * @fn
 * 配列を行列として表示する
 * @param (int m) 行列の行数
 * @param (int n) 行列の列数
 * @param (const float * x) 表示する配列
 * @return 無し
 */
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

/**
 * @fn
 * 全結合層の計算をする y = Ax + b
 * @param (int m) 行列Aの行数であり，出力ベクトルの要素数
 * @param (int n) 行列Aの列数であり，入力ベクトルの要素数
 * @param (const flaot * x) 入力ベクトルの配列
 * @param (const float * A) 全結合層の重みパラメータ(行列)の配列
 * @param (const float * b) 全結合層のバイアスパラメータ(ベクトル)の配列
 * @param (float * y) 出力ベクトルの配列
 * @return 無し
 * @detail yは念の為，0で初期化しているが，果たして必要か否か
 */
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

/**
 * @fn
 * 活性化関数ReLuを計算する
 * @param (int m) ベクトルの要素数
 * @param (const flaot * x) 入力ベクトルの配列
 * @param (float * y) 出力ベクトルの配列
 * @return 無し
 */
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

/**
 * @fn
 * 活性化関数softmaxを計算する
 * @param (int m)
 * @param ()
 * @param ()
 * @return 無し
 * @detail
 */
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
float loss(const float * y, int t) {
    return - log(y[t] + 1e-7);
}
void save(const char * filename, int m, int n, const float * A, const float * b) {
    FILE * fp;
    fp = fopen(filename, "w");

    int i;
    int j;
    for (i = 0; i < m; i ++) {
        for (j = 0; j < n; j ++) {
            fprintf(fp, "%f ", A[n * i + j]);
        }
        fprintf(fp, "\n");
    }

    for (i = 0; i < m; i ++) {
        fprintf(fp, "%f\n", b[i]);
    }
    fclose(fp);
}
void load(const char * filename, int m, int n, float * A, float * b) {
    FILE * fp;
	if ((fp = fopen(filename, "r")) == NULL) {
		printf("file open error!!\n");
		exit(EXIT_FAILURE);
	}

    int i;
    int j;
    for (i = 0; i < m; i ++) {
        for (j = 0; j < n; j ++) {
            fscanf(fp, "%f", &A[n * i + j]);
        }
    }

    for (i = 0; i < m; i ++) {
        fscanf(fp, "%f", &b[i]);
    }
    fclose(fp);
}

double normal_rand(float mu, float sigma) {
    float x = ((float)rand() + 1.0) / ((float)RAND_MAX + 2.0);
    float y = ((float)rand() + 1.0) / ((float)RAND_MAX + 2.0);
    float z = sqrt(- 2.0*log(x)) * sin(2.0*M_PI*y);
    return mu + sigma*z;
}
void normal_rand_init(int n, unsigned seed, float * o) {
    srand(seed);
    float sigma = sqrt(2 / (float)n);
    int i;
    for (i = 0; i < n; i ++) {
        o[i] = normal_rand(0, sigma);
    }
}

void progress(float x) {
    int count = (int)(x / 0.1);
    printf("[");
    int i;
    for (i = 0; i < count - 1; i ++) {
        printf("=");
    }
    printf(">");
    for (i = 0; i < 10 - count; i ++) {
        printf(" ");
    }
    printf("] %.1f%% ", x * 100.0);
}

void batch_normalization(int m, float * x) {
    float avg = 0;
    float e = 10e-5;
    int i;
    for (i = 0; i < m; i ++) {
        avg += x[i];
    }
    avg /= m;
    float sigma = 0;;
    for (i = 0; i < m; i ++) {
        sigma += pow((x[i] - avg), 2);
    }
    sigma /= m;
    for (i = 0; i < m; i ++) {
        x[i] = (x[i] - avg) / sqrt(sigma + e);
    }
}
