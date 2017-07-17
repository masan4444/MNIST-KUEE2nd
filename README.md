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
### `nnlib.c`の関数
#### 1．行列の表示 `print`
```c
void print(int m, int n, const float * x)
```
配列を`m * n`行列として表示する．
#### 2．配列同士の足し算 `add`
```c
void add(int n, const float * x, float * o)
```
要素数が`n`である配列同士の足し算を行う．配列`o`の要素が`x`に足される．
#### 3．配列同士の引き算 `sub`
```c
void sub(int n, const float * x, const float * y, float * o)
```
要素が`n`である配列同士の引き算を行う．配列`o`に`x`から`y`を引いた結果が代入される形となる．`MomentumSGD`で使用される．
#### 4．配列の定数倍 `scale`
```c
void scale(int n, float x, float * o)
```
要素数が`n`の配列`o`の各要素に`x`をかける．
#### 5．配列を定数倍し，別の配列に足す `scale_and_add`
```c
void scale_and_add(int n, float x, const float * y, float * o)
```
要素数が`n`の配列`y`の各要素に`x`をかけ，その結果を`o`に足す．
#### 6．配列の初期化 `init`
```c
void init(int n, float x, float * o)
```
要素数が`n`の配列`o`の各要素を`x`で初期化する．
#### 7．配列を乱数で初期化 `rand_init`
```c
void rand_init(int n, unsigned seed, float * o)
```
要素数が`n`の配列`o`を`- 1`から`1`の一様乱数で初期化する．`seed`は`rand()`のシード値となる．
#### 8．全結合層の計算 `fc`
```c
void fc(int m, int n, const float * x, const float * A, const float * b, float * y)
```
サイズが`n`の入力ベクトル`x`を受け取り，サイズが`m`の出力ベクトル`y`を計算する．`A`は全結合層の重みパラメータであり，`m * n`の行列を配列に格納したものである．`b`はバイアスパラメータであり，サイズが`m`のベクトルである．
#### 9．活性化関数ReLu `relu`
```c
void relu(int m, const float * x, float * y)
```
## 拡張・改善した点
