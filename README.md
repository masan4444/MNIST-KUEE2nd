# 最終課題
###### 氏名:
###### 学籍番号:

## プログラム概要
ニューラルネットワークを実装し，学習させるため4つのソースコードを作成した．
* `nnlib.c`
ニューラルネットワークに共通で使われる関数をまとめたライブラリ

* `nnlib.h`
`nnlib.c`のヘッダーファイル

* `mnist3layer.c`
3層ニューラルネットワーク ``(fc -> relu -> softmax)`` の学習．推論を行うプログラム

* `mnistNlayer.c`
6層以上のニューラルネットワーク ``((fc -> relu) × N -> fc -> softmax)`` の学習．推論を行うプログラム


## プログラムについて
### 学習のさせ方
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
### 推論の仕方
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
### プログラムの処理の流れ
追記予定



## 関数の説明
(注記) 「`float`型でサイズが`m`の配列`x`」と書いてあるものは，「ポインタ`x`の以後`sizeof(float) * m`byte分に`m`個の`float`型の数値が存在する」と仮定し，それを保証する．確保されているサイズがそれ以下の場合，不具合をおこす．
### `nnlib.c`の関数
#### 1．行列の表示 `print`
```c
void print(int m, int n, const float * x)
```
サイズが`m * n`の配列`x`を`m * n`行列として表示する．
#### 2．配列同士の足し算 `add`
```c
void add(int n, const float * x, float * o)
```
サイズが`n`である配列同士の足し算を行う．配列`o`の要素が`x`に足される．
#### 3．配列同士の引き算 `sub`
```c
void sub(int n, const float * x, const float * y, float * o)
```
サイズが`n`である配列同士の引き算を行う．配列`o`に`x`から`y`を引いた結果が代入される形となる．`MomentumSGD`で使用される．
#### 4．配列の定数倍 `scale`
```c
void scale(int n, float x, float * o)
```
サイズが`n`の配列`o`の各要素に`x`をかける．
#### 5．配列を定数倍し，別の配列に足す `scale_and_add`
```c
void scale_and_add(int n, float x, const float * y, float * o)
```
サイズが`n`の配列`y`の各要素に`x`をかけ，その結果を`o`に足す．
#### 6．配列の初期化 `init`
```c
void init(int n, float x, float * o)
```
サイズが`n`の配列`o`の各要素を`x`で初期化する．
#### 7．配列を乱数で初期化 `rand_init`
```c
void rand_init(int n, unsigned seed, float * o)
```
サイズが`n`の配列`o`を`- 1`から`1`の一様乱数で初期化する．`seed`は`rand()`のシード値となる．
#### 8．全結合層の計算 `fc`
```c
void fc(int m, int n, const float * x, const float * A, const float * b, float * y)
```
サイズが`n`の入力ベクトルを受け取り，サイズが`m`の出力ベクトルを計算する．`x`が入力ベクトルの配列であり，`y`が出力ベクトルの配列である．`A`は全結合層の重みパラメータであり，`m * n`の行列を配列に格納したものである．`b`はバイアスパラメータであり，要素数がが`m`のベクトルを配列に格納したもの．．
#### 9．活性化関数ReLu `relu`
```c
void relu(int m, const float * x, float * y)
```
要素数が`m`のベクトルを配列に格納したもの`x`を受け取り，ReLuの計算結果をに`y`代入する．
#### 10．活性化関数Softmax `softmax`
```c
void softmax(int m, const float * x, float * y)
```
要素数が`m`のベクトルを配列に格納したもの`x`を受け取り，Softmaxの計算結果をに`y`代入する．
#### 11．Softmaxと損失関数の誤差逆伝播 `softmaxwithloss_bwd`
```c
void softmaxwithloss_bwd(int m, const float * y, unsigned char t, float * dx)
```
`m`はベクトルのサイズであり，出力ベクトルの配列`y`，正解ラベル`t`を用いて，Softmaxと損失関数の偏微分を計算し，`dx`に格納する．
#### 12．ReLuの誤差逆伝播 `relu_bwd`
```c
void relu_bwd(int m, const float * x, const float * dy, float * dx)
```
`m`はベクトルのサイズであり，入力ベクトルの配列`x`，下流の偏微分の配列`dy`を用いて，ReLuの偏微分を計算し，`dx`に格納する
#### 13．全結合層(fc)の誤差逆伝播 `fc_bwd`
```c
void fc_bwd(int m, int n, const float * x, const float * dy, const float * A, float * dA, float * db, float * dx)
```
入力ベクトルの配列`x`，下流の偏微分の配列`dy`，重みパラメータ行列の配列`A`を用いて，Aの偏微分`dA`，bの偏微分`db`，上流に流す偏微分`dx`を計算する．`dA`は`m * n`の行列を配列に格納したもの，`db`は要素数`m`のベクトルを配列に格納したもの，`dx`は要素数が`n`のベクトルを配列に格納したものになる．
#### 14．配列のシャッフル `shuffle`
```c
void shuffle(int n, int * x, unsigned seed)
```
サイズが`n`の配列`x`の各要素をシャッフルする関数．`seed`は`rand()`のシード値で，`seed`を変えることで，シャッフルの結果も変化し，同じなら同じように並び替える．
#### 15．交差エントロピー誤差 `cross_entropy_error`
```c
float cross_entropy_error(const float * y, int t)
```
ニューラルネットワークの出力ベクトルが`y`，正解ラベルが`t`のときの交差エントロピー誤差を計算する．
#### 16
## 拡張・改善した点
