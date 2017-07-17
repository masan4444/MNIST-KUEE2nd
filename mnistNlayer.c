/**
 * @file mnist3layer.c
 * @brief (fc -> relu) * (layer_num - 1) -> fc -> softmaxの，(layer_num * 2)層構造のニューラルネットワークの学習と推論をするプログラム
 * @author masan4444
 * @date 2017/07/15
 * @detail
 * 学習：「./実行ファイル名 train {SGD, MomentumSGD} パラメータを保存するファイル名からファイル形式を除いたもの ファイル形式」で，指定した勾配法で学習し，指定したファイル名でパラメータを保存する
 * 推論：「./実行ファイル名 パラメータが保存されているファイル名からファイル形式を除いたもの ファイル形式 画像ファイル名」で推論できる
 * 例として，「./a.out train SGD A dat」とすると，SGDで学習し，学習したパラメータをA1.dat, A2.dat, A3.dat...として保存する．．
 */

#include <string.h>
#include "nn.h"
#include "nnlib.h"

#define FLOAT_SIZE sizeof(float)

#define layer_num 3 //層の総数としてはlayer_num * 2となる
#define m0 50 //1つ目の全結合層の出力ベクトルの要素数
#define m1 100 //2つ目の全結合層の出力ベクトルの要素数

//SGDのハイパーパラメータ
#define epoch_SGD 30
#define batch_size_SGD 100
#define initial_learning_rate_SGD 0.5

//MomentumSGDのハイパーパラメータ
#define epoch_MomentumsGD 30
#define batch_size_MomentumSGD 100
#define learning_rate_MomentumSGD 0.04
#define momentum_MomentumSGD 0.9

void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename_without_formatname, const char * formatname);
void MomentumSGD(int epoch, int batch_size, float learning_rate, float momentum, const char * filename_without_formatname, const char * formatname);

void inferenceMode(const char * filename_without_formatname, const char * formatname, const char * bmp_filename);
int inference(const float ** A, const float ** b,  const float * x, float * y);
void backward(const float ** A, const float ** b, const float * x, unsigned char t, float * y, float ** dA, float ** db);
void accRate_and_loss(const float ** A, const float ** b, const float * test_x, const unsigned char * test_y, float * accRate, float * Loss);

void saveAll(const char * filename_without_formatname, const char * formatname, float ** A, float ** b);

int m[layer_num] = {m0, m1, 10};
int n[layer_num] = {784, m0, m1};
//それぞれの全結合層の重みパラメータ，行列Aのサイズ

int main(int argc, char const * argv[]) {
    if (!strcmp(argv[1], "train")) {
        if (!strcmp(argv[2], "SGD")) {
            SGD(epoch_SGD, batch_size_SGD, initial_learning_rate_SGD, argv[3], argv[4]);
        } else if (!strcmp(argv[2], "MomentumSGD")) {
            MomentumSGD(epoch_MomentumsGD, batch_size_MomentumSGD, learning_rate_MomentumSGD, momentum_MomentumSGD, argv[3], argv[4]);
        }
    } else if (!strcmp(argv[1], "inference")) {
        inferenceMode(argv[2], argv[3], argv[4]);
        // (filename_without_formatname, formatname, bmp_filename),
    }
    return 0;
}

/**
 * @fn
 * 確率的勾配法を用いて学習する．ただしエッポクが進むごとに学習率を変化させている．
 * @param (int epoch) エッポクの総数
 * @param (int batch_size) batchのサイズ
 * @param (float initial_learning_rate) 最初の学習率
 * @param (const char * filename_without_formatname) 学習したパラメータを保存するファイル名．ただし拡張子は含まない．
 * @param (const char * formatname) パラメータを保存するファイルの拡張子
 * @return 無し
 * @detail エッポクごとに学習率を減少させていくことで効率的に学習している．具体的にはエッポク数nの時の学習率を「最初の学習率 / n」としている．
 */
void SGD(int epoch, int batch_size, float initial_learning_rate, const char * filename_without_formatname, const char * formatname) {
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

    float * A[layer_num];
    float * b[layer_num];
    float * dA[layer_num];
    float * db[layer_num];
    float * dA_sum[layer_num];
    float * db_sum[layer_num];

    int i;
    for (i = 0; i < layer_num; i ++) {
        A[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        normal_rand_init(m[i]*n[i], 1, A[i]);
        b[i] = malloc(FLOAT_SIZE*m[i]);
        normal_rand_init(m[i], 2, b[i]);

        dA[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db[i] = malloc(FLOAT_SIZE*m[i]);
        dA_sum[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        db_sum[i] = malloc(FLOAT_SIZE*m[i]);
    }

    float * y = malloc(FLOAT_SIZE*10);

    int * index = malloc(sizeof(int)*60000);
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer:SGD layer_num:%d\nepoch:%d, batch_size:%d, initial_learning_rate:%f\n", layer_num, epoch, batch_size, initial_learning_rate);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        float learning_rate = initial_learning_rate / (epoch_time + 1);
        //float learning_rate = initial_learning_rate / (epoch_time / 10 + 1);
        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");

            int k;
            for (k = 0; k < layer_num; k ++) {
                init(m[k]*n[k], 0, dA_sum[k]);
                init(m[k], 0, db_sum[k]);
            }
            for (j = 0; j < batch_size; j ++) {
                for (k = 0; k < layer_num; k ++) {
                    init(m[k]*n[k], 0, dA[k]);
                    init(m[k], 0, db[k]);
                }
                //init(10, 0, y);
                backward((const float **)A, (const float **)b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);
                for (k = 0; k < layer_num; k ++) {
                    add(m[k]*n[k], dA[k], dA_sum[k]);
                    add(m[k], db[k], db_sum[k]);
                }
            }
            for (k = 0; k < layer_num; k ++) {
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

/**
 * @fn
 * 確率的勾配降下法を用いて学習する．ただし更新に慣性項と呼ばれるものを付与している．
 * @param (int epoch) エッポクの総数
 * @param (int batch_size) batchのサイズ
 * @param (float learning_rate) 学習率
 * @param (float momentum) どれだけ慣性を付与するかのハイパーパラメータ
 * @param (const char * filename_without_formatname) 学習したパラメータを保存するファイル名．ただし拡張子は含まない．
 * @param (const char * formatname) パラメータを保存するファイルの拡張子
 * @return 無し
 * @detail パラメータ更新の際に，前回のパラメータの更新量にmomentumをかけた量を追加することで，学習をより慣性的なものにしている．
 */
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

    float * A[layer_num];
    float * b[layer_num];
    float * dA[layer_num];
    float * db[layer_num];
    float * dA_sum[layer_num];
    float * db_sum[layer_num];

    float * A_last[layer_num];
    float * b_last[layer_num];
    float * A_diff[layer_num];
    float * b_diff[layer_num];

    int i;
    for (i = 0; i < layer_num; i ++) {
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

    float * y = malloc(FLOAT_SIZE*10);

    int * index = malloc(sizeof(int)*60000);
    for (i = 0; i < 60000; i ++) {
        index[i] = i;
    }

    int j;
    int epoch_time;

    printf("optimizer:MomentumSGD layer_num:%d\nepoch:%d batch_size:%d learning_rate:%f momentum:%f\n", layer_num, epoch, batch_size, learning_rate, momentum);

    for (epoch_time = 0; epoch_time < epoch; epoch_time ++) {
        shuffle(60000, index, epoch_time); //epoch_time as seed
        //int learning_rate = initial_learning_rate / (epoch_time + 1);

        for (i = 0; i < 60000/batch_size; i ++) {
            printf("epoch:%d ", epoch_time + 1);
            progress((float)i / (float)(60000/batch_size));
            printf("\r");
            int k;
            for (k = 0; k < layer_num; k ++) {
                init(m[k]*n[k], 0, dA_sum[k]);
                init(m[k], 0, db_sum[k]);
            }
            for (j = 0; j < batch_size; j ++) {
                for (k = 0; k < layer_num; k ++) {
                    init(m[k]*n[k], 0, dA[k]);
                    init(m[k], 0, db[k]);
                }
                //init(10, 0, y);
                backward((const float **)A, (const float **)b, train_x + index[i*batch_size + j] * 784, train_y[index[i*batch_size + j]], y, dA, db);
                for (k = 0; k < layer_num; k ++) {
                    add(m[k]*n[k], dA[k], dA_sum[k]);
                    add(m[k], db[k], db_sum[k]);
                }

            }
            for (k = 0; k < layer_num; k ++) {
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

/**
 * @fn
 * (fc -> relu) * (layer_num - 1) -> fc -> softmaxの，(layer_num * 2)層構造のニューラルネットワークで，与えられた画像に書かれている数字を推論し，表示する．
 * @param (const char * filename_without_formatname) 学習したパラメータを保存しているファイル名．ただし拡張子は含まない．
 * @param (const char * formatname) パラメータを保存しているファイルの拡張子
 * @param (const char * bmp_filename) 数字が書かれている画像ファイル名
 * @return 無し
 */
void inferenceMode(const char * filename_without_formatname, const char * formatname, const char * bmp_filename) {
    float * A[layer_num];
    float * b[layer_num];

    int i;
    for (i = 0; i < layer_num; i ++) {
        A[i] = malloc(FLOAT_SIZE*m[i]*n[i]);
        b[i] = malloc(FLOAT_SIZE*m[i]);
    }
    for (i = 0; i < layer_num; i ++) {
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

/**
 * @fn
 * (fc -> relu) * (layer_num - 1) -> fc -> softmax の(layer_num * 2)層構造のニューラルネットワークで推論する
 * @param (const float ** A) 重みパラメータ，行列Aの配列の配列．i番目の全結合層の重みパラメータ行列がA[i]である．
 * @param (const float ** b) バイアスパラメータ，ベクトルbの配列の配列．i番目の全結合層のバイアスパラメータベクトルがb[i]である．
 * @param (const float * x) 入力ベクトルの配列
 * @param (float * y) 出力ベクトルの配列
 * @return ネットワークによって推論される数字(0 ~ 9)
 * @detail 引数に出力ベクトルの配列を追加することで，下記のaccRate_and_loss関数のおいて，正解率と損失関数を同時に求めている．
 */
int inference(const float ** A, const float ** b, const float * x, float * y) {
    float * x_relu[layer_num - 1];
    float * x_fc[layer_num];

    x_fc[0] = malloc(FLOAT_SIZE*n[0]);
    memcpy(x_fc[0], x, FLOAT_SIZE*n[0]);

    int i;
    for (i = 0; i < layer_num - 1; i ++) {
        x_relu[i] =  malloc(FLOAT_SIZE*m[i]);
        fc(m[i], n[i], x_fc[i], A[i], b[i], x_relu[i]);
        x_fc[i + 1] = malloc(FLOAT_SIZE*m[i]);
        relu(m[i], x_relu[i], x_fc[i + 1]);
    }

    float * x_softmax = malloc(FLOAT_SIZE*m[layer_num - 1]);
    fc(m[layer_num - 1], n[layer_num - 1], x_fc[layer_num - 1], A[layer_num - 1], b[layer_num - 1], x_softmax);
    softmax(m[layer_num - 1], x_softmax, y);

    float max = y[0];
    int ans = 0;
    for (i = 1; i <= 9; i ++) {
        if (y[i] > max) {
            max = y[i];
            ans = i;
        }
    }

    for (i = 0; i < layer_num - 1; i ++) {
        free(x_fc[i]);
        free(x_relu[i]);
    }
    free(x_fc[layer_num - 1]);
    free(x_softmax);

    return ans;
}

/**
 * @fn
 * (fc -> relu) * (layer_num - 1) -> fc -> softmax の(layer_num * 2)層構造のニューラルネットワークを，誤差逆伝播法を用いて勾配を求める
 * @param (const float ** A) 重みパラメータ，行列Aの配列の配列．i番目の全結合層の重みパラメータ行列がA[i]である．
 * @param (const float ** b) バイアスパラメータ，ベクトルbの配列の配列．i番目の全結合層のバイアスパラメータベクトルがb[i]である．
 * @param (const float * x) 入力ベクトルの配列
 * @param (unsigned char t) 正解ラベル(0 ~ 9)
 * @param (float * y) 出力ベクトルの配列
 * @param (float ** dA) 重みパラメータAの勾配の配列の配列．i番目の全結合層の重みパラメータの勾配がdA[i]である．
 * @param (float ** db) バイアスパラメータbの勾配の配列の配列．i番目の全結合層のバイアスパラメータの勾配がdb[i]である．
 * @return 無し
 */
void backward(const float ** A, const float ** b, const float * x, unsigned char t, float * y, float ** dA, float ** db) {

    float * x_relu[layer_num - 1];
    float * x_fc[layer_num];

    x_fc[0] = malloc(FLOAT_SIZE*n[0]);
    memcpy(x_fc[0], x, FLOAT_SIZE*n[0]);
    //batch_normalization(n[0], x_fc[0]); //////

    int i;
    for (i = 0; i < layer_num - 1; i ++) {
        x_relu[i] =  malloc(FLOAT_SIZE*m[i]);
        //batch_normalization(n[i], x_fc[i]); //////
        fc(m[i], n[i], x_fc[i], A[i], b[i], x_relu[i]);
        x_fc[i + 1] = malloc(FLOAT_SIZE*m[i]);
        relu(m[i], x_relu[i], x_fc[i + 1]);
    }

    float * x_softmax = malloc(FLOAT_SIZE*m[layer_num - 1]);
    //batch_normalization(n[2], x_fc[2]); //////
    fc(m[layer_num - 1], n[layer_num - 1], x_fc[layer_num - 1], A[layer_num - 1], b[layer_num - 1], x_softmax);
    softmax(m[layer_num - 1], x_softmax, y);

    float * dx_relu[layer_num - 1];
    float * dx_fc[layer_num];
    for (i = 0; i < layer_num - 1; i ++) {
        dx_relu[i] = malloc(FLOAT_SIZE*m[i]);
        dx_fc[i] = malloc(FLOAT_SIZE*n[i]);
    }
    dx_fc[layer_num - 1] = malloc(FLOAT_SIZE*n[layer_num - 1]);
    float * dx_softmax = malloc(FLOAT_SIZE*m[layer_num - 1]);

    softmaxwithloss_bwd(m[layer_num - 1], y, t, dx_softmax);
    fc_bwd(m[layer_num - 1], n[layer_num - 1], x_fc[layer_num - 1], dx_softmax, A[layer_num - 1], dA[layer_num - 1], db[layer_num - 1], dx_fc[layer_num - 1]);

    for (i = layer_num - 2; i >= 0; i --) {
        relu_bwd(m[i], x_relu[i], dx_fc[i + 1], dx_relu[i]);
        fc_bwd(m[i], n[i], x_fc[i], dx_relu[i], A[i], dA[i], db[i], dx_fc[i]);
    }

    for (i = 0; i < layer_num - 1; i ++) {
        free(x_fc[i]);
        free(x_relu[i]);
        //free(dx_fc[i]);
        //free(dx_relu[i]);
    }
    free(x_softmax);
    //free(x_fc[layer_num - 1]);
    //free(dx_fc[layer_num - 1]);
    //free(dx_softmax);

}

/**
 * @fn
 * (fc -> relu) * (layer_num - 1) -> fc -> softmax の(layer_num * 2)層構造のニューラルネットワークに対して，テストデータの正解率と損失関数を表示する
 * @param (const float ** A) 重みパラメータ，行列Aの配列の配列．i番目の全結合層の重みパラメータ行列がA[i]である．
 * @param (const float ** b) バイアスパラメータ，ベクトルbの配列の配列．i番目の全結合層のバイアスパラメータベクトルがb[i]である．
 * @param (float * y) 出力ベクトルの配列
 * @param (const float * test_x) 入力ベクトル(テストデータ)をすべて集めた配列
 * @param (const unsigned char * test_y) 正解ラベル(テストデータ)をすべて集めた配列
 * @param (float * accRate) 正解率を格納するポインタ
 * @param (float * Loss) 損失関数の値を格納するポインタ
 * @return 無し
 * @detail inference関数で出力ベクトルを求めることで，正解率と損失関数を同時に求めるようにしている．そのため正解率と損失関数の値はポインタで取得している．
 */
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

/**
 * @fn
 * 学習したパラメータを保存する
 * @param (const char * filename_without_formatname) 学習したパラメータを保存するファイル名．ただし拡張子は含まない．
 * @param (const char * formatname) パラメータを保存するファイルの拡張子
 * @param (const float ** A) 重みパラメータ，行列Aの配列の配列．i番目の全結合層の重みパラメータ行列がA[i]である
 * @param (const float ** b) バイアスパラメータ，ベクトルbの配列の配列．i番目の全結合層のバイアスパラメータベクトルがb[i]である．
 * @return 無し
 * @detail 例としてsaveAll(X, dat, A, b)とすると，A[0],b[0]をA1.datに，A[1],b[1]をA2.datに保存する．
 */
void saveAll(const char * filename_without_formatname, const char * formatname, float ** A, float ** b) {
    int i;
    for (i = 0; i < layer_num; i ++) {
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
