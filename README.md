# 最終課題
###### 氏名:
###### 学籍番号:

## 1．プログラム概要
### 1．ソースコードについて
ニューラルネットワークを実装し，学習させるため4つのソースコードを作成した．
* `nnlib.c`
ニューラルネットワークに共通で使われる関数をまとめたライブラリ

* `nnlib.h`
`nnlib.c`のヘッダーファイル

* `mnist3layer.c`
3層ニューラルネットワーク ``(fc -> relu -> softmax)`` の学習．推論を行うプログラム

* `mnistNlayer.c`
6層以上のニューラルネットワーク ``((fc -> relu) × N -> fc -> softmax)`` の学習．推論を行うプログラム

### 2．コンパイル
3層ニューラルネットワーク`mnist3layer.c`の場合
```Shell
gcc -Wall mnist3layer.c nnlib.c -lm
```
6層以上のニューラルネットワーク(N層)`mnistNlayer.c`の場合
```Shell
gcc -Wall mnistNlayer.c nnlib.c -lm
```
### 3．学習の仕方
3層ニューラルネットワーク`mnist3layer.c`の場合
```
./a.out train {optimizerの指定} {学習したパラメータを保存するファイル名}
```
6層以上のニューラルネットワーク(N層)`mnistNlayer.c`の場合
```
./a.out train {optimizerの指定} {学習したパラメータを保存するファイル名から拡張子を除いたもの} {拡張子の名前}
```
とすることで，学習できる．ただし，`optimiser`は`SGD`，または`MomentumSGD`が用意されている．(後述)

例として，N層の場合
```
./a.out train SGD A dat
```
とすることで，`A1.dat`というファイルに一つ目の全結合層(fc層)のパラメータを，`A2.dat`に2つ目の全結合層のパラメータを保存する．N層の場合，合計`N / 2`個の全結合層があるのでファイルも同じ数だけ生成される．
### 4．推論の仕方
3層ニューラルネットワーク`mnist3layer.c`の場合
```
./a.out inference {パラメータが保存されているファイル名} {数字が書かれているbmp画像ファイルの名前}
```
6層以上のニューラルネットワーク(N層)`mnistNlayer.c`の場合
```
./a.out inference {パラメータが保存されているファイル名から拡張子を除いたもの} {拡張子の名前} {数字が書かれているbmp画像ファイルの名前}
```
とすることで，与えられた画像に書かれている数字を推論する．

例として，N層の場合
```
./a.out inference A dat number.bmp
```
とすると，`A?.dat`を用いて，`number.bmp`に書かれている数字を推論する．
### 5．プログラムの大まかな処理の流れ`SGD`
1. マクロにおいて，optimizerのハイパーパラメータを設定する．(17 ~ 31行目)
2. メモリの割り当てる(86 ~ 93行目)
3. 正規分布に従った乱数によって初期化(95 ~ 96行目)
4. ランダムシャッフル用の配列`index`を作成し，0 ~ 59999で初期化する．(98 ~ 102行目)
5. 配列`index`を`shuffle`によってランダムシャッフルする．(110行目)
6. 進捗状況の表示をする．(113 ~ 115行目)
7. 平均勾配`dA_sum`,`db_sum`の初期化をする．(117 ~ 118行目)
8. バッチサイズ分の勾配を`backward3`を用いて求め．`dA_sum`,`db_sum`に足していく．この際`index`を用いて，バッチに用いる訓練データがランダムになるようにする．(119 ~ 126行目)
9. `dA_sum`,`db_sum`をバッチサイズで割ったもの平均勾配として，これを用いて`A`,`b`を更新する
10. 5 ~ 8の操作を訓練データ / バッチサイズ分繰り返す．
11. テストデータを用いて，学習したパラメータの正解率と損失を計算，表示する．
12. パラメータを指定されたファイルに保存する．
13. 4 ~ 12の操作を1エッポクとして，マクロで指定したエッポク数分繰り返す．

## 2．関数の説明
(注記) 「`float`型でサイズが`m`の配列`x`」と書いてあるものは，「ポインタ`x`の以後`sizeof(float) * m`byte分に`m`個の`float`型の数値が存在する」と仮定し，それを保証する．確保されているサイズがそれ以下の場合，不具合をおこす．
### 1．`nnlib.c`の関数
#### 1-1．行列の表示 `print`
```c
void print(int m, int n, const float * x)
```
サイズが`m * n`の配列`x`を`m * n`行列として表示する．
#### 1-2．配列同士の足し算 `add`
```c
void add(int n, const float * x, float * o)
```
サイズが`n`である配列同士の足し算を行う．配列`o`の要素が`x`に足される．
#### 1-3．配列同士の引き算 `sub`
```c
void sub(int n, const float * x, const float * y, float * o)
```
サイズが`n`である配列同士の引き算を行う．配列`o`に`x`から`y`を引いた結果が代入される形となる．`MomentumSGD`で使用される．
#### 1-4．配列の定数倍 `scale`
```c
void scale(int n, float x, float * o)
```
サイズが`n`の配列`o`の各要素に`x`をかける．
#### 1-5．配列を定数倍し，別の配列に足す `scale_and_add`
```c
void scale_and_add(int n, float x, const float * y, float * o)
```
サイズが`n`の配列`y`の各要素に`x`をかけ，その結果を`o`に足す．
#### 1-6．配列の初期化 `init`
```c
void init(int n, float x, float * o)
```
サイズが`n`の配列`o`の各要素を`x`で初期化する．
#### 1-7．配列を乱数で初期化 `rand_init`
```c
void rand_init(int n, unsigned seed, float * o)
```
サイズが`n`の配列`o`を`- 1`から`1`の一様乱数で初期化する．`seed`は`rand()`のシード値となる．
#### 1-8．全結合層の計算 `fc`
```c
void fc(int m, int n, const float * x, const float * A, const float * b, float * y)
```
サイズが`n`の入力ベクトルを受け取り，サイズが`m`の出力ベクトルを計算する．`x`が入力ベクトルの配列であり，`y`が出力ベクトルの配列である．`A`は全結合層の重みパラメータであり，`m * n`の行列を配列に格納したものである．`b`はバイアスパラメータであり，要素数がが`m`のベクトルを配列に格納したもの．
#### 1-9．活性化関数ReLU `relu`
```c
void relu(int m, const float * x, float * y)
```
要素数が`m`のベクトルを配列に格納したもの`x`を受け取り，ReLUの計算結果をに`y`代入する．
#### 1-10．活性化関数Softmax `softmax`
```c
void softmax(int m, const float * x, float * y)
```
要素数が`m`のベクトルを配列に格納したもの`x`を受け取り，Softmaxの計算結果をに`y`代入する．
#### 1-11．Softmaxと損失関数の誤差逆伝播 `softmaxwithloss_bwd`
```c
void softmaxwithloss_bwd(int m, const float * y, unsigned char t, float * dx)
```
`m`はベクトルのサイズであり，出力ベクトルの配列`y`，正解ラベル`t`を用いて，Softmaxと損失関数の偏微分を計算し，`dx`に格納する．
#### 1-12．ReLUの誤差逆伝播 `relu_bwd`
```c
void relu_bwd(int m, const float * x, const float * dy, float * dx)
```
`m`はベクトルのサイズであり，入力ベクトルの配列`x`，下流の偏微分の配列`dy`を用いて，ReLUの偏微分を計算し，`dx`に格納する
#### 1-13．全結合層(fc)の誤差逆伝播 `fc_bwd`
```c
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx)
```
入力ベクトルの配列`x`，下流の偏微分の配列`dy`，重みパラメータ行列の配列`A`を用いて，Aの偏微分`dA`，bの偏微分`db`，上流に流す偏微分`dx`を計算する．`dA`は`m * n`の行列を配列に格納したもの，`db`は要素数`m`のベクトルを配列に格納したもの，`dx`は要素数が`n`のベクトルを配列に格納したものになる．
#### 1-14．配列のシャッフル `shuffle`
```c
void shuffle(int n, int * x, unsigned seed)
```
サイズが`n`の配列`x`の各要素をシャッフルする関数．`seed`は`rand()`のシード値で，`seed`を変えることで，シャッフルの結果も変化し，同じなら同じように並び替える．
#### 1-15．交差エントロピー誤差 `cross_entropy_error`
```c
float cross_entropy_error(const float * y, int t)
```
ニューラルネットワークの出力ベクトルが`y`，正解ラベルが`t`のときの交差エントロピー誤差を計算する．
#### 1-16．パラメータの保存 `save`
```c
void save(const char * filename, int m, int n, const float * A, const float * b)
```
`m * n`行列の重みパラメータ`A`,`m`行ベクトルのバイアスパラメータ`b`を`char`型配列`filename`に格納されているファイル名に保存する．
#### 1-17．パラメータの読込 `load`
```c
void load(const char * filename, int m, int n, float * A, float * b)
```
`m * n`行列の重みパラメータ`A`,`m`行ベクトルのバイアスパラメータ`b`を`char`型配列`filename`に格納されているファイル名から読み込む．
#### 1-18．正規分布に従った乱数を生成 `normal_rand`
```c
float normal_rand(float mu, float sigma)
```
平均`mu`，分散`sigma`の正規分布に従った`float`型の乱数を生成し，返り値で返す．
#### 1-19．サイズが`n`の配列を平均0，分散sqrt(2/n)の正規分布に従った乱数で初期化 `normal_rand_init`
```c
void normal_rand_init(int n, unsigned seed, float * o)
```
サイズが`n`の配列`o`を，平均が`0`，分散が`sqrt(2 / n)`の正規分布に従った乱数で初期化する．`seed`は`rand()`のシード値．
#### 1-20．進捗状況の表示 `progress`
```c
void progress(float x)
```
進捗状況`x`(0から1までの値)をグラフとして表示する．例として，`x`が0.753のとき
```bash
[======>   ] 75.3%
```
のように表示する．
### 2．`mnist3layer.c`の関数
#### 2-1．行列の表示 `print`
```c
void print(int m, int n, const float * x)
```
サイズが`m * n`の配列`x`を`m * n`行列として表示する．
## 拡張・改善した点
