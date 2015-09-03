#ifndef MEFEATURE_H_INCLUDED
#define MEFEATURE_H_INCLUDED

#include <iostream>
#include <cmath>
#include <cstring>

/* Maximum Entropy Model （最大エントロピーモデル）の1つの素性を表現するクラス */
class MEFeature {
private:
  int N_gram;            /* 素性のグラム数 */
  int *pattern_x;        /* Nグラムの1つ前までの文字列. N_gram-1の長さの配列（単語は整数と対応付けている）
		            pattern_x[0]=x_1, patern_x[1]=x_2, ... , pattern_x[N_gram-2]=x_{N_gram-1} */
  int pattern_y;         /* Nグラムの今の単語 */

public:
  double weight;         /* 素性の重み */
  int    count;          /* 素性のパターンが学習データで表れた回数 */
  double empirical_prob; /* 素性の学習データにおける経験確率 */
  double empirical_E;    /* この素性の経験期待値 */
  double model_E;        /* モデルの期待値 */
  double parameter;      /* 素性に付随する, モデルのパラメタ */
  bool   is_marginal;    /* この素性が周辺素性（yのみに依存して活性化する素性）か否（条件付き素性; xにも依存して決まる素性）か */
  bool   is_additive;    /* この素性は素性総和のために追加された素性か否か */

  /* コンストラクタ. */
  MEFeature(int N_gram, int *pattern_x, int pattern_y, int count=1, double weight=1.0f, bool is_additive=false);
  /* コピーコンストラクタ.(Vectorを使うので..) */
  MEFeature(const MEFeature &src);
  /* デストラクタ. pattern_xの解放 */
  ~MEFeature(void);

  /* 以下, メソッド */
private:
  /* メンバのコピールーチン */
  MEFeature& copy(const MEFeature &src);

public:
  /* 仮引数のパターンtest_x,test_yに対してこの素性が活性化しているか調べ,
     活性化していたらweightを返し, 活性化していなければ0を返す */
  double checkget_weight(int xlength, int *test_x, int test_y);
  /* 仮引数のパターンに対して活性化しているか調べ,
     活性化していればweight*parameterを返す. （エネルギー関数; exp内部計算用）*/
  double checkget_param_weight(int xlength, int *test_x, int test_y);
  /* パターンチェックのサブルーチン. 活性化していればtrue, していなければfalse */
  bool   check_pattern(int xlength, int *test_x, int test_y);
  /* 素性情報を表示する. */
  void print_info(void);

};

#endif /* MEFEATURE_H_INCLUDED */
