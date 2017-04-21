
#include <string.h>
#include "nn.h"
#include "nnlib.h"


int inference3(const float * A, const float * b, const float * x);
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dA, float * db);
float acc_rate(const float * A, const float * b, const float * test_x, const unsigned char * test_y);

int main() {
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = - 1;

    float * test_x = NULL;
    unsigned char * test_y = NULL;
    int test_count = - 1;

    int width = - 1;
    int height = - 1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);
/*
    float A[6] = {1, 3, 0, 2, 4, 1};
    float x[3] = {5, 2, 3};
    float dy[2] = {3, 5};
    float * dx = malloc(sizeof(float)*3);
    float * dA = malloc(sizeof(float)*6);
    float * db = malloc(sizeof(float)*2);

    fc_bwd(2, 3, x, dy, A, dA, db, dx);
    print(2, 3, dA);
    print(2, 1, db); 

    int epoch = 10;
    int batch_size = 100;
    float learning_rate = 0.01;
*/
    float * y = malloc(sizeof(float)*10);
    //float * A = malloc(sizeof(float)*784*10);
    //float * b = malloc(sizeof(float)*10);
    float * dA = malloc(sizeof(float)*784*10);
    float * db = malloc(sizeof(float)*10);
    backward3(A_784x10, b_784x10, train_x + 784*8, train_y[8], y, dA, db);
    //print(784, 1, train_x + 784*8);
    
/*
    float * dA_sum = malloc(sizeof(float)*784*10);
    float * db_sum = malloc(sizeof(float)*10);

    rand_init(784*10, 44, A);
    rand_init(10, 45, b);

    print(1, 10, b);

    int * index = malloc(sizeof(int)*60000);
    int i;
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }
    shuffle(60000, index, 88);
    int j;
    int epoch_time;
    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        for (i = 0; i < 60000/batch_size; i ++) {
            init(784*10, 0, dA_sum);
            init(10, 0, db_sum);
            for (j = 0; j < batch_size; j ++) {
                init(784*10, 0, dA);
                init(10, 0, db);
                backward3(A, b, train_x + index[i*batch_size + j] * 784, train_y[i*batch_size + j], y, dA, db);
                add(784*10, dA, dA_sum);
                add(10, db, db_sum);
            }
            scale(784*10, - learning_rate / batch_size, dA_sum);
            add(784*10, dA_sum, A);
            scale(10, - learning_rate / batch_size, db_sum);
            add(10, db_sum, b);
        }
        printf("%f\n", acc_rate(A, b, test_x, test_y));
    }
    */
    return 0;
}

int inference3(const float * A, const float * b, const float * x) {
    float * y = malloc(sizeof(float)*10);
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    float max = y[0];
    int ans = 0;
    int i;
    for (i = 1; i < 10; i ++) {
        if (y[i] > max) {
            max = y[i];
            ans = i;
        }
    }
    return ans;
}

void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dA, float * db) {
    fc(10, 784, x, A, b, y);
    float * x_relu = malloc(sizeof(float)*10);
    memcpy(x_relu, y, sizeof(float)*10);
    relu(10, x_relu, y);
    float * x_softmax = malloc(sizeof(float)*10);
    memcpy(x_softmax, y, sizeof(float)*10);
    softmax(10, x_softmax, y);

    float * dx = malloc(sizeof(float)*10);
    softmaxwithloss_bwd(10, y, t, dx);
    relu_bwd(10, x_relu, dx, dx);
    float * dx_fc = malloc(sizeof(float)*784*10);
    fc_bwd(10, 784, x, dx, A, dA, db, dx_fc);
}

float acc_rate(const float * A, const float * b, const float * test_x, const unsigned char * test_y) {
    int sum = 0;
    int i;
    for (i = 0; i < 10000; i ++) {
        if (inference3(A, b, test_x + i*784) == test_y[i]) {
            sum ++;
        }
    }
    return sum * 100.0 / 10000;
}
