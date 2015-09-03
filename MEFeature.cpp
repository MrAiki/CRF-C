#include "MEFeature.hpp"

/* コンストラクタ. */
MEFeature::MEFeature(int N_gram, int *pattern_x, int pattern_y, int count, double weight, bool is_additive)
{
  /* Nグラムのサイズのセット */
  this->N_gram    = N_gram;

  /* ユニグラム（前の単語に注目しない）以上ならば, pattern_xをセットする. */
  if (N_gram > 1) {
    this->pattern_x = new int[N_gram-1];
    memcpy(this->pattern_x, pattern_x, sizeof(int) * (N_gram-1));
  } else if (N_gram == 1) {
    this->pattern_x = NULL;
  }
  
  /* 今の単語, 重みのセット */
  this->pattern_y   = pattern_y;
  this->weight      = weight;
  this->is_additive = is_additive;
  this->count       = count;

  /* TODO:パラメタは何で初期化しよう？
     parameter = 0;
  */
}

/* コピーコンストラクタ */
MEFeature::MEFeature(const MEFeature &src)
{
  this->copy(src);
}

/* デストラクタ */
MEFeature::~MEFeature(void) 
{
  if (pattern_x != NULL && N_gram > 1) {
    delete [] pattern_x;
  }
} 

/* オブジェクトのコピールーチン */
MEFeature& MEFeature::copy(const MEFeature &src)
{
  /* 自分自身を引数に入れていた時は, 自分を返す */
  if (this == &src) return *this;

  /* メンバをコピー */
  this->N_gram    = src.N_gram;
  this->pattern_y = src.pattern_y;
  if (N_gram > 1) {
    this->pattern_x = new int[N_gram-1];
    memcpy(this->pattern_x, src.pattern_x, sizeof(int) * (N_gram-1));
  }
  this->weight         = src.weight;
  this->count          = src.count;
  this->empirical_prob = src.empirical_prob;
  this->empirical_E    = src.empirical_E;
  this->model_E        = src.model_E;
  this->parameter      = src.parameter;
  this->is_marginal    = src.is_marginal;
  this->is_additive    = src.is_additive;

  return *this;
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

/* 素性情報を表示する */
void MEFeature::print_info(void)
{
  std::cout << N_gram << "-gram model" << std::endl;
  std::cout << "Pattern X: ";
  if (N_gram > 1) {
    for (int i = 0; i < (N_gram-1); i++) {
      std::cout << "x[" << i << "]:" << pattern_x[i] << " ";
    }
  } else {
    std::cout << "(nothing; uni-gram.)";
  }
  std::cout << std::endl;

  std::cout << "Pattern Y: " << pattern_y << std::endl;
  std::cout << "parameter: " << parameter << std::endl;
  std::cout << "weight: " << weight << std::endl;
  std::cout << "frequency count: " << count << std::endl;
  std::cout << "empirical prob.: " << empirical_prob << std::endl;
  std::cout << "empirical avg.: " << empirical_E << std::endl;
  std::cout << "model avg.: " << model_E << std::endl;
  std::cout << "marginal feature?: " << is_marginal << std::endl;
  std::cout << "additive feature(f_[n+1])? " << is_additive << std::endl;
}
