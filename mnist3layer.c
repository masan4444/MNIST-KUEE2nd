#include <string.h>
#include "nn.h"
#include "nnlib.h"

#define FLOAT_SIZE sizeof(float)

void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename);
void MomentumSGD(int epoch, int batch_size, float learning_rate, float momentum, const char * filename);

int inference3(const float * A, const float * b, const float * x, float * y);
void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dA, float * db);
void accRate_and_loss(const float * A, const float * b, const float * test_x, const unsigned char * test_y, float * accRate, float * Loss);
void inferenceMode(const char * filename, const char * bmp_filename);

int main(int argc, char const * argv[]) {
    if (!strcmp(argv[1], "train")) {
        //SGD(50, 100, 1, argv[2]);
        //(epoch, batch_size, initial_learning_rate, filename)
        MomentumSGD(10, 100, 0.01, 0.1, "S.dat");
        //(epoch, batch_size, learning_rate, momentum, filename)

    } else if (!strcmp(argv[1], "inference")) {
        inferenceMode(argv[2], argv[3]);
        //(filename, bmp_filenam),
    }
    return 0;
}

/**
 * @fn
 * 確率的勾配法を用いて学習する
 * @param (int epoch) エッポク数
 * @param (int batch_size) rand()のシード値
 * @param (float * o) 初期化する配列
 * @return 無し
 */
void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename) {
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

    float * y = malloc(FLOAT_SIZE*10);
    float * A = malloc(FLOAT_SIZE*784*10);
    float * b = malloc(FLOAT_SIZE*10);
    float * dA = malloc(FLOAT_SIZE*784*10);
    float * db = malloc(FLOAT_SIZE*10);

    float * dA_sum = malloc(FLOAT_SIZE*784*10);
    float * db_sum = malloc(FLOAT_SIZE*10);

    rand_init(784*10, 44, A);
    rand_init(10, 45, b);

    int * index = malloc(sizeof(int)*60000);
    int i;
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer=SGD\nepoch=%d, batch_size=%d, initial_learning_rate=%f\n", epoch, batch_size, initial_learning_rate);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        float learning_rate = initial_learning_rate / (epoch_time  + 1);
        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");

            init(784*10, 0, dA_sum);
            init(10, 0, db_sum);
            for (j = 0; j < batch_size; j ++) {
                init(784*10, 0, dA);
                init(10, 0, db);
                init(10, 0, y);
                backward3(A, b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);
                add(784*10, dA, dA_sum);
                add(10, db, db_sum);
            }
            scale_and_add(784*10, -learning_rate / batch_size, dA_sum, A);
            scale_and_add(10, -learning_rate / batch_size, db_sum, b);
        }
        float acc_rate;
        float Loss;
        accRate_and_loss(A, b, test_x, test_y, &acc_rate, &Loss);
        printf("epoch:%d acc_rate:%.2f%% loss:%.2f learning_rate:%.4f\n", epoch_time + 1, acc_rate, Loss, learning_rate);

        save(filename, 10, 784, A, b);
   }
}

void MomentumSGD(int epoch, int batch_size, float learning_rate, float momentum, const char * filename) {
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = - 1;

    float * test_x = NULL;
    unsigned char * test_y = NULL;
    int test_count = - 1;

    int width = - 1;
    int height = - 13;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);

    float * A = malloc(FLOAT_SIZE*10*784);
    normal_rand_init(10*784, 1, A);
    float * b = malloc(FLOAT_SIZE*10);
    normal_rand_init(10, 2, b);

    float * dA = malloc(FLOAT_SIZE*10*784);
    float * db = malloc(FLOAT_SIZE*10);
    float * dA_sum = malloc(FLOAT_SIZE*10*784);
    float * db_sum = malloc(FLOAT_SIZE*10);

    float * A_last = malloc(FLOAT_SIZE*10*784);
    float * b_last = malloc(FLOAT_SIZE*10);
    float * A_diff = malloc(FLOAT_SIZE*10*784);
    float * b_diff = malloc(FLOAT_SIZE*10);
    init(10*784, 0, A_diff);
    init(10, 0, b_diff);

    float * y = malloc(FLOAT_SIZE*10);

    int i;
    int * index = malloc(sizeof(int)*60000);
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer:MomentumSGD layer_num:%d\nepoch:%d batch_size:%d learning_rate:%f momentum:%f\n", 1, epoch, batch_size, learning_rate, momentum);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        //int learning_rate = initial_learning_rate / (epoch_time + 1);

        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");

            init(10*784, 0, dA_sum);
            init(10, 0, db_sum);

            for (j = 0; j < batch_size; j ++) {
                init(10*784, 0, dA);
                init(10, 0, db);
                //init(10, 0, y);
                backward3(A, b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);

                add(10*784, dA, dA_sum);
                add(10, db, db_sum);
            }

            memcpy(A_last, A, FLOAT_SIZE*10*784);
            scale_and_add(10*784, -learning_rate / batch_size, dA_sum, A);
            scale_and_add(10*784, momentum, A_diff, A);
            sub(10*784, A, A_last, A_diff);

            memcpy(b_last, b, FLOAT_SIZE*10);
            scale_and_add(10, -learning_rate / batch_size, db_sum, b);
            scale_and_add(10, momentum, b_diff, b);
            sub(10, b, b_last, b_diff);
        }

        float acc_rate;
        float Loss;
        accRate_and_loss(A, b, test_x, test_y, &acc_rate, &Loss);
        printf("epoch:%d acc_rate:%.2f%% loss:%.2f\n", epoch_time + 1, acc_rate, Loss);

        save(filename, 10, 784, A, b);
    }
}

int inference3(const float * A, const float * b, const float * x, float * y) {
    init(10, 0, y);
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    float max = y[0];
    int ans = 0;
    int i;
    for (i = 1; i <= 9; i ++) {
        if (y[i] > max) {
            max = y[i];
            ans = i;
        }
    }
    return ans;
}

void backward3(const float * A, const float * b, const float * x, unsigned char t, float * y, float * dA, float * db) {
    fc(10, 784, x, A, b, y);
    float * x_relu = malloc(FLOAT_SIZE*10);
    memcpy(x_relu, y, FLOAT_SIZE*10);
    relu(10, x_relu, y);
    float * x_softmax = malloc(FLOAT_SIZE*10);
    memcpy(x_softmax, y, FLOAT_SIZE*10);
    softmax(10, x_softmax, y);

    float * dx = malloc(FLOAT_SIZE*10);
    init(10, 0, dx);
    softmaxwithloss_bwd(10, y, t, dx);
    relu_bwd(10, x_relu, dx, dx);
    float * dx_fc = malloc(FLOAT_SIZE*784);
    init(784, 0, dx_fc);
    fc_bwd(10, 784, x, dx, A, dA, db, dx_fc);
    free(dx_fc);
}

void accRate_and_loss(const float * A, const float * b, const float * test_x, const unsigned char * test_y, float * accRate, float * Loss) {
    int sum_acc = 0;
    float sum_loss = 0;
    int i;
    for (i = 0; i < 10000; i ++) {
        float * y = malloc(FLOAT_SIZE*10);
        if (inference3(A, b, test_x + i*784, y) == test_y[i]) {
            sum_acc ++;
        }
        sum_loss += cross_entropy_error(y, test_y[i]);
    }
    *accRate = sum_acc * 100.0 / 10000;
    *Loss = sum_loss / 10000;
}

void inferenceMode(const char * filename, const char * bmp_filename) {
    float * A = malloc(FLOAT_SIZE*10*784);
    float * b = malloc(FLOAT_SIZE*10);

    load(filename, 10, 784, A, b);

    float * x = load_mnist_bmp(bmp_filename);
    float * y = malloc(FLOAT_SIZE*10);
    printf("%d\n", inference3(A, b, x, y));
}
