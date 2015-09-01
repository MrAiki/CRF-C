#include "MEFeature.hpp"

/* コンストラクタ. */
MEFeature::MEFeature(int N_gram, int *pattern_x, int pattern_y, double weight, bool is_additive)
{
  /* Nグラムのサイズのセット */
  this->N_gram    = N_gram;

  /* ユニグラム（前の単語に注目しない）以上ならば, pattern_xをセットする. */
  if (N_gram > 1) {
    this->pattern_x = new int[N_gram-1];
    memcpy(this->pattern_x, pattern_x, sizeof(int) * (N_gram-1));
  }
  
  /* 今の単語, 重みのセット */
  this->pattern_y   = pattern_y;
  this->weight      = weight;
  this->is_additive = is_additive;
}
  
/* デストラクタ */
MEFeature::~MEFeature(void) 
{
  if (pattern_x != NULL) {
    delete pattern_x;
  }
}
  
/* パターンチェックのサブルーチン. 
   活性化していればtrue, していなければfalseを返す */
bool MEFeature::check_pattern(int xlength, int *test_x, int test_y)
{
  /* 想定するxのパターン長さがこの素性以上, あるいはyのパターンが一致しなけばfalse */
  if (xlength > N_gram || test_y != pattern_y) {
    return false;
  }

  /* この素性のパターン長さが1（ユニグラム）ならば一致が確認できたのでtrue */
  if (N_gram == 1) {
    return true;
  }

  /* xのパターンを走査してチェック. */
  for (int i=0; i < (N_gram-1); i++) {
    if (test_x[i] != pattern_x[i]) {
      return false;
    }
  }

  /* 全ての一致が確認できたのでtrue */
  return true;
}

/* パターンに対し活性化しているか調べ, 
   活性化していればweightを返し, していなければ0を返す */
double MEFeature::checkget_weight(int xlength, int *test_x, int test_y)
{
  if (check_pattern(xlength, test_x, test_y)) {
    return weight; 
  } else {
    return 0.0f;
  }
}

/* パターンに対し活性化しているか調べ, 
   活性化していればparameter*weightを返し, していなければ0を返す */
double MEFeature::checkget_param_weight(int xlength, int *test_x, int test_y)
{
  if (check_pattern(xlength, test_x, test_y)) {
    return (parameter * weight);
  } else {
    return 0.0f;
  }
}
