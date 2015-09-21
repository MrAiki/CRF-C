#include "MEFeature.hpp"

/* コンストラクタ. */
MEFeature::MEFeature(int N_gram, std::vector<int> pattern_x, int pattern_y, int count, double weight)
{
  /* Nグラムのサイズのセット */
  this->N_gram = N_gram;

  /* ユニグラム（前の単語に注目しない）以上ならば, pattern_xをセットする. */
  if (N_gram > 1) {
    this->pattern_x.resize(pattern_x.size());
    std::copy(pattern_x.begin(), pattern_x.end(), this->pattern_x.begin());
  } else {
    pattern_x.clear();
  }

  /* 今の単語, 重みのセット */
  this->pattern_y   = pattern_y;
  this->weight      = weight;
  this->count       = count;

  /* その他メンバを適当に初期化 */
  this->empirical_prob = 0.0f;
  this->empirical_E    = 0.0f;
  this->model_E        = 0.0f;
  this->is_marginal    = false;
  this->parameter      = 0.0f;  /* パラメタの初期値は要審議 */
  
}

/* コピーコンストラクタ */
MEFeature::MEFeature(const MEFeature &src)
{
  this->copy(src);
}

/* デフォルトコンストラクタ */
MEFeature::MEFeature(void)
{
  pattern_x.clear();
}

/* デストラクタ */
MEFeature::~MEFeature(void) { ; } 

/* オブジェクトのコピールーチン */
MEFeature& MEFeature::copy(const MEFeature &src)
{
  /* 自分自身を引数に入れていた時は, 自分を返す */
  if (this == &src) return *this;

  /* メンバをコピー */
  this->N_gram    = src.N_gram;
  this->pattern_y = src.pattern_y;
  if (N_gram > 1) {
    this->pattern_x.resize(src.pattern_x.size());
    std::copy(src.pattern_x.begin(), src.pattern_x.end(), this->pattern_x.begin());
  } else {
    pattern_x.clear();
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

/* パターンの取得ルーチン */
int MEFeature::get_N_gram(void)
{
  return N_gram;
}

std::vector<int> MEFeature::get_pattern_x(void)
{
  return pattern_x;
}

int MEFeature::get_pattern_y(void)
{
  return pattern_y;
}
  
/* パターンチェックのサブルーチン. 
   活性化していればtrue, していなければfalseを返す */
bool MEFeature::check_pattern(std::vector<int> test_x, int test_y)
{
  /* テストするxのパターンの長さがこの素性以下, あるいはyのパターンが一致しなけばfalse */
  if (test_x.size() < (unsigned int)(N_gram-1) || test_y != pattern_y) {
    return false;
  }

  /* この素性のパターン長さが1（ユニグラム）ならば一致が確認できたのでtrue */
  if (N_gram == 1) {
    return true;
  }

  /* xのパターンを走査してチェック. */
  for (int i=0; i < (N_gram-1); i++) {
    /* test_xは末尾の(N_gram-1)文字のみを比較する */
    if (test_x[test_x.size()-(N_gram-1)+i] != pattern_x[i]) {
      return false;
    }
  }

  /* 全ての一致が確認できたのでtrue */
  return true;
}

/* パターンに対し活性化しているか調べ, 
   活性化していればweightを返し, していなければ0を返す */
double MEFeature::checkget_weight(std::vector<int> test_x, int test_y)
{
  if (check_pattern(test_x, test_y)) {
    return weight; 
  } else {
    return 0.0f;
  }
}

/* パターンに対し活性化しているか調べ, 
   活性化していればparameter*weightを返し, していなければ0を返す */
double MEFeature::checkget_param_weight(std::vector<int> test_x, int test_y)
{
  if (check_pattern(test_x, test_y)) {
    return (parameter * weight);
  } else {
    return 0.0f;
  }
}

/* 仮引数のパターンに対して活性化しているか調べ,
   活性化していればweight*empirical_probを返す. （経験期待値計算用） */
double MEFeature::checkget_weight_emprob(std::vector<int> test_x, int test_y) 
{
  if (check_pattern(test_x, test_y)) {
    return (weight * empirical_prob);
  } else {
    return 0.0f;
  }
}

/* パターンの完全一致を確かめるサブルーチン. */
bool MEFeature::strict_check_pattern(std::vector<int> test_x, int test_y)
{
  /* 長さもチェックする */
  if (test_x.size() == (unsigned int)(N_gram-1)
      && check_pattern(test_x, test_y)) {
    return true;
  } else {
    return false;
  }
}


/* 素性情報を表示する */
void MEFeature::print_info(void)
{
  std::cout << N_gram << "-gram model feature" << std::endl;
  std::cout << "Pattern X: ";
  if (N_gram > 1) {
    for (int i = 0; i < (N_gram-1); i++) {
      std::cout << "x[" << i << "]:" << pattern_x[i] << " ";
    }
  } else {
    std::cout << "(nothing == uni-gram.)";
  }
  std::cout << std::endl;

  std::cout << "Pattern Y: " << pattern_y << std::endl;
  std::cout << "Parameter: " << parameter << std::endl;
  std::cout << "Weight: " << weight << std::endl;
  std::cout << "Frequency count: " << count << std::endl;
  std::cout << "Empirical prob.: " << empirical_prob << std::endl;
  std::cout << "Empirical avg.: " << empirical_E << std::endl;
  std::cout << "Model avg.: " << model_E << std::endl;
  std::cout << "Marginal feature?: ";
  if (is_marginal) {
    std::cout << "Yes" << std::endl;
  } else {
    std::cout << "No" << std::endl;
  }
}
