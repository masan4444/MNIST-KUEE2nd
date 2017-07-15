#include <string.h>
#include "nn.h"
#include "nnlib.h"

#define FLOAT_SIZE sizeof(float)

#define num_of_layer 3 //3以上ならOK
#define m0 50
#define m1 100
#define m2 30

void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename_without_formatname, const char * formatname);
void MomentumSGD(int epoch, int batch_size, float learning_rate, float momentum, const char * filename_without_formatname, const char * formatname);

void inferenceMode(const char * filename_without_formatname, const char * formatname, const char * bmp_filename);
int inference(const float ** A, const float ** b,  const float * x, float * y);
void backward(const float ** A, const float ** b, const float * x, unsigned char t, float * y, float ** dA, float ** db);
void accRate_and_loss(const float ** A, const float ** b, const float * test_x, const unsigned char * test_y, float * accRate, float * Loss);

void saveAll(const char * filename_without_formatname, const char * formatname, float ** A, float ** b);

/*
int * m = malloc(sizeof(int)*num_of_layer);
int * n = malloc(sizeof(int)*num_of_layer);
m[0] = m0;
m[1] = m1;
m[num_of_layer - 1] = 10;
n[0] = 784;
n[1] = m0;
n[num_of_layer - 1] = m1;
*/

int m[num_of_layer] = {m0, m1, 10};
int n[num_of_layer] = {784, m0, m1};

int main(int argc, char const * argv[]) {
    if (!strcmp(argv[1], "train")) {
        //SGD(30, 100, 0.5, argv[2], argv[3]);
        // (epoch, batch_size, filename_without_formatname, formatname)
        MomentumSGD(30, 100, 001, 0.1, argv[2], argv[3]);
    } else if (!strcmp(argv[1], "inference")) {
        inferenceMode(argv[2], argv[3], argv[4]);
        // (filename_without_formatname, formatname, bmp_filename),
    }
    return 0;
}

void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename_without_formatname, const char * formatname) {
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

    float * A[num_of_layer];
    float * b[num_of_layer];
    float * dA[num_of_layer];
    float * db[num_of_layer];
    float * dA_sum[num_of_layer];
    float * db_sum[num_of_layer];

    int i;
    for (i = 0; i < num_of_layer; i ++) {
        A[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        normal_rand_init(m[i]*n[i], 1, A[i]);
        b[i] = malloc(FLOAT_SIZE*m[i]);
        normal_rand_init(m[i], 2, b[i]);

        dA[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db[i] = malloc(FLOAT_SIZE*m[i]);
        dA_sum[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db_sum[i] = malloc(FLOAT_SIZE*m[i]);
    }
    /*
    memcpy(A[0], A1_784_50_100_10, FLOAT_SIZE*m[0]*n[0]);
    memcpy(b[0], b1_784_50_100_10, FLOAT_SIZE*m[0]);
    memcpy(A[1], A2_784_50_100_10, FLOAT_SIZE*m[1]*n[1]);
    memcpy(b[1], b2_784_50_100_10, FLOAT_SIZE*m[1]);
    memcpy(A[2], A3_784_50_100_10, FLOAT_SIZE*m[2]*n[2]);
    memcpy(b[2], b3_784_50_100_10, FLOAT_SIZE*m[2]);
    */

    float * y = malloc(FLOAT_SIZE*10);

    int * index = malloc(sizeof(int)*60000);
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer:SGD num_of_layer:%d\nepoch:%d, batch_size:%d, initial_learning_rate:%f\n", num_of_layer, epoch, batch_size, initial_learning_rate);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        float learning_rate = initial_learning_rate / (epoch_time + 1);
        //float learning_rate = initial_learning_rate / (epoch_time / 10 + 1);
        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");

            int k;
            for (k = 0; k < num_of_layer; k ++) {
                init(m[k]*n[k], 0, dA_sum[k]);
                init(m[k], 0, db_sum[k]);
            }
            for (j = 0; j < batch_size; j ++) {
                for (k = 0; k < num_of_layer; k ++) {
                    init(m[k]*n[k], 0, dA[k]);
                    init(m[k], 0, db[k]);
                }
                //init(10, 0, y);
                backward((const float **)A, (const float **)b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);
                for (k = 0; k < num_of_layer; k ++) {
                    add(m[k]*n[k], dA[k], dA_sum[k]);
                    add(m[k], db[k], db_sum[k]);
                }
            }
            for (k = 0; k < num_of_layer; k ++) {
                scale_and_add(m[k]*n[k], -learning_rate / batch_size, dA_sum[k], A[k]);
                scale_and_add(m[k], -learning_rate / batch_size, db_sum[k], b[k]);
            }
        }
        float acc_rate;
        float Loss;
        accRate_and_loss((const float **)A, (const float **)b, test_x, test_y, &acc_rate, &Loss);
        printf("epoch:%d acc_rate:%.2f%% loss:%.2f learning_rate:%.2f\n", epoch_time + 1, acc_rate, Loss, learning_rate);

        saveAll(filename_without_formatname, formatname, A, b);
    }
}

void MomentumSGD(int epoch, int batch_size, float learning_rate, float momentum, const char * filename_without_formatname, const char * formatname) {
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

    //int epoch = 20;
    //int batch_size = 100;
    //float initial_learning_rate = 0.05;

    float * A[num_of_layer];
    float * b[num_of_layer];
    float * dA[num_of_layer];
    float * db[num_of_layer];
    float * dA_sum[num_of_layer];
    float * db_sum[num_of_layer];

    float * A_last[num_of_layer];
    float * b_last[num_of_layer];
    float * A_diff[num_of_layer];
    float * b_diff[num_of_layer];

    int i;
    for (i = 0; i < num_of_layer; i ++) {
        A[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        normal_rand_init(m[i]*n[i], 1, A[i]);
        b[i] = malloc(FLOAT_SIZE*m[i]);
        normal_rand_init(m[i], 2, b[i]);

        dA[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db[i] = malloc(FLOAT_SIZE*m[i]);
        dA_sum[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db_sum[i] = malloc(FLOAT_SIZE*m[i]);

        A_last[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        b_last[i] = malloc(FLOAT_SIZE*m[i]);
        A_diff[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        b_diff[i] = malloc(FLOAT_SIZE*m[i]);
        init(m[i]*n[i], 0, A_diff[i]);
        init(m[i], 0, b_diff[i]);
    }
    /*
    memcpy(A[0], A1_784_50_100_10, FLOAT_SIZE*m[0]*n[0]);
    memcpy(b[0], b1_784_50_100_10, FLOAT_SIZE*m[0]);
    memcpy(A[1], A2_784_50_100_10, FLOAT_SIZE*m[1]*n[1]);
    memcpy(b[1], b2_784_50_100_10, FLOAT_SIZE*m[1]);
    memcpy(A[2], A3_784_50_100_10, FLOAT_SIZE*m[2]*n[2]);
    memcpy(b[2], b3_784_50_100_10, FLOAT_SIZE*m[2]);
    */

    float * y = malloc(FLOAT_SIZE*10);

    int * index = malloc(sizeof(int)*60000);
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer:MomentumSGD num_of_layer:%d\nepoch:%d batch_size:%d learning_rate:%f momentum:%f\n", num_of_layer, epoch, batch_size, learning_rate, momentum);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        //int learning_rate = initial_learning_rate / (epoch_time + 1);

        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");
            int k;
            for (k = 0; k < num_of_layer; k ++) {
                init(m[k]*n[k], 0, dA_sum[k]);
                init(m[k], 0, db_sum[k]);
            }
            for (j = 0; j < batch_size; j ++) {
                for (k = 0; k < num_of_layer; k ++) {
                    init(m[k]*n[k], 0, dA[k]);
                    init(m[k], 0, db[k]);
                }
                //init(10, 0, y);
                backward((const float **)A, (const float **)b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);
                for (k = 0; k < num_of_layer; k ++) {
                    add(m[k]*n[k], dA[k], dA_sum[k]);
                    add(m[k], db[k], db_sum[k]);
                }

            }
            for (k = 0; k < num_of_layer; k ++) {
                memcpy(A_last[k], A[k], FLOAT_SIZE*m[k]*n[k]);
                scale_and_add(m[k]*n[k], -learning_rate / batch_size, dA_sum[k], A[k]);
                scale_and_add(m[k]*n[k], momentum, A_diff[k], A[k]);
                sub(m[k]*n[k], A[k], A_last[k], A_diff[k]);

                memcpy(b_last[k], b[k], FLOAT_SIZE*m[k]);
                scale_and_add(m[k], -learning_rate / batch_size, db_sum[k], b[k]);
                scale_and_add(m[k], momentum, b_diff[k], b[k]);
                sub(m[k], b[k], b_last[k], b_diff[k]);
            }
        }
        float acc_rate;
        float Loss;
        accRate_and_loss((const float **)A, (const float **)b, test_x, test_y, &acc_rate, &Loss);
        printf("epoch:%d acc_rate:%.2f%% loss:%.2f\n", epoch_time + 1, acc_rate, Loss);

        saveAll(filename_without_formatname, formatname, A, b);
    }
}

void inferenceMode(const char * filename_without_formatname, const char * formatname, const char * bmp_filename) {
    float * A[num_of_layer];
    float * b[num_of_layer];

    int i;
    for (i = 0; i < num_of_layer; i ++) {
        A[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        b[i] = malloc(FLOAT_SIZE*m[i]);
    }
    for (i = 0; i < num_of_layer; i ++) {
        char filename[8];
        strcpy(filename, filename_without_formatname);
        char str_i[8];
        sprintf(str_i, "%d", i + 1);
        strcat(str_i, ".");
        strcat(str_i, formatname);
        strcat(filename, str_i);
        load(filename, m[i], n[i], A[i], b[i]);
    }
    float * x = load_mnist_bmp(bmp_filename);
    float * y = malloc(FLOAT_SIZE*10);
    printf("%d\n", inference((const float **)A, (const float **)b, x, y));
}

int inference(const float ** A, const float ** b, const float * x, float * y) {

    float * x_relu[num_of_layer - 1];
    float * x_fc[num_of_layer];

    x_fc[0] = malloc(FLOAT_SIZE*n[0]);
    memcpy(x_fc[0], x, FLOAT_SIZE*n[0]);

    int i;
    for (i = 0; i < num_of_layer - 1; i ++) {
        x_relu[i] =  malloc(FLOAT_SIZE*m[i]);
        fc(m[i], n[i], x_fc[i], A[i], b[i], x_relu[i]);
        x_fc[i + 1] = malloc(FLOAT_SIZE*m[i]);
        relu(m[i], x_relu[i], x_fc[i + 1]);
    }

    float * x_softmax = malloc(FLOAT_SIZE*m[num_of_layer - 1]);
    fc(m[num_of_layer - 1], n[num_of_layer - 1], x_fc[num_of_layer - 1], A[num_of_layer - 1], b[num_of_layer - 1], x_softmax);
    softmax(m[num_of_layer - 1], x_softmax, y);

    float max = y[0];
    int ans = 0;
    for (i = 1; i <= 9; i ++) {
        if (y[i] > max) {
            max = y[i];
            ans = i;
        }
    }

    for (i = 0; i < num_of_layer - 1; i ++) {
        free(x_fc[i]);
        free(x_relu[i]);
    }
    free(x_fc[num_of_layer - 1]);
    free(x_softmax);

    return ans;
}

void backward(const float ** A, const float ** b, const float * x, unsigned char t, float * y, float ** dA, float ** db) {

    float * x_relu[num_of_layer - 1];
    float * x_fc[num_of_layer];

    x_fc[0] = malloc(FLOAT_SIZE*n[0]);
    memcpy(x_fc[0], x, FLOAT_SIZE*n[0]);
    //batch_normalization(n[0], x_fc[0]); //////

    int i;
    for (i = 0; i < num_of_layer - 1; i ++) {
        x_relu[i] =  malloc(FLOAT_SIZE*m[i]);
        //batch_normalization(n[i], x_fc[i]); //////
        fc(m[i], n[i], x_fc[i], A[i], b[i], x_relu[i]);
        x_fc[i + 1] = malloc(FLOAT_SIZE*m[i]);
        relu(m[i], x_relu[i], x_fc[i + 1]);
    }

    float * x_softmax = malloc(FLOAT_SIZE*m[num_of_layer - 1]);
    //batch_normalization(n[2], x_fc[2]); //////
    fc(m[num_of_layer - 1], n[num_of_layer - 1], x_fc[num_of_layer - 1], A[num_of_layer - 1], b[num_of_layer - 1], x_softmax);
    softmax(m[num_of_layer - 1], x_softmax, y);

    float * dx_relu[num_of_layer - 1];
    float * dx_fc[num_of_layer];
    for (i = 0; i < num_of_layer - 1; i ++) {
        dx_relu[i] = malloc(FLOAT_SIZE*m[i]);
        dx_fc[i] = malloc(FLOAT_SIZE*n[i]);
    }
    dx_fc[num_of_layer - 1] = malloc(FLOAT_SIZE*n[num_of_layer - 1]);
    float * dx_softmax = malloc(FLOAT_SIZE*m[num_of_layer - 1]);

    softmaxwithloss_bwd(m[num_of_layer - 1], y, t, dx_softmax);
    fc_bwd(m[num_of_layer - 1], n[num_of_layer - 1], x_fc[num_of_layer - 1], dx_softmax, A[num_of_layer - 1], dA[num_of_layer - 1], db[num_of_layer - 1], dx_fc[num_of_layer - 1]);

    for (i = num_of_layer - 2; i >= 0; i --) {
        relu_bwd(m[i], x_relu[i], dx_fc[i + 1], dx_relu[i]);
        fc_bwd(m[i], n[i], x_fc[i], dx_relu[i], A[i], dA[i], db[i], dx_fc[i]);
    }

    for (i = 0; i < num_of_layer - 1; i ++) {
        free(x_fc[i]);
        free(x_relu[i]);
        //free(dx_fc[i]);
        //free(dx_relu[i]);
    }
    free(x_softmax);
    //free(x_fc[num_of_layer - 1]);
    //free(dx_fc[num_of_layer - 1]);
    //free(dx_softmax);

}

void accRate_and_loss(const float ** A, const float ** b, const float * test_x, const unsigned char * test_y, float * accRate, float * Loss) {
    int sum_acc = 0;
    float sum_loss = 0;
    int i;
    for (i = 0; i < 10000; i ++) {
        float * y = malloc(FLOAT_SIZE*10);
        if (inference(A, b, test_x + i*784, y) == test_y[i]) {
            sum_acc ++;
        }
        sum_loss += cross_entropy_error(y, test_y[i]);
    }
    *accRate = sum_acc * 100.0 / 10000;
    *Loss = sum_loss / 10000;
}

void saveAll(const char * filename_without_formatname, const char * formatname, float ** A, float ** b) {
    int i;
    for (i = 0; i < num_of_layer; i ++) {
        char filename[8];
        strcpy(filename, filename_without_formatname);
        char str_i[8];
        sprintf(str_i, "%d", i + 1);
        strcat(str_i, ".");
        strcat(str_i, formatname);
        strcat(filename, str_i);
        save(filename, m[0], n[0], A[0], b[0]);
    }
}
