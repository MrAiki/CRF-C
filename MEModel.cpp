#include "MEModel.hpp"

/* コンストラクタ */
MEModel::MEModel(int maxN_gram, int pattern_count_bias,
                 int max_iteration_learn=DEFAULT_MAX_ITERATION_LEARN, double epsilon_learn=DEFAULT_EPSILON_LEARN,
		 int max_iteration_f_select=DEFAULT_MAX_F_SIZE, double epsilon_f_select=DEFAULT_EPSILON_F_SELECTION,
		 int max_iteration_f_gain=DEFAULT_MAX_ITERATION_FGAIN, double epsilon_f_gain=DEFAULT_EPSILON_FGAIN)
{
  this->maxN_gram              = maxN_gram;
  this->pattern_count_bias     = pattern_count_bias;
  this->max_iteration_learn    = max_iteration_learn;
  this->epsilon_learn          = epsilon_learn;
  this->max_iteration_f_select = max_iteration_f_select;
  this->epsilon_f_select       = epsilon_f_select;
  this->max_iteration_f_gain   = max_iteration_f_gain;
  this->epsilon_f_gain         = epsilon_f_gain;
}

/* デストラクタ. 分布と正規化項の解放 */
MEModel::~MEModel(void)
{
  delete joint_prob; delete cond_prob;
  /* TODO:サイズが分からない　delete norm_factor; */
}

/* 現在読み込み中のファイルから次の単語を返すサブルーチン */
std::string MEModel::next_word(void)
{
  char ch;          /* 現在読んでいる文字 */
  std::string ret;  /* 返り値の文字列 */

  /* 改行している場合は一行読む */
  if (line_index == -1) {
    if (input_file && std::getline(input_file, line_buffer)) {
      /* 注：std::getlineは改行文字をバッファに格納しない */
      line_index = 0; /* インデックスは0にリセット */
    } else {
      ret += EOF;
      return ret;     /* ファイルの終端に達している時はEOFを返す */
    }
  }

  /* 空白, タブ文字の読み飛ばし */
  ch = line_buffer[line_index++];
  while (ch == ' ' || ch == '\t')
    ch = line_buffer[line_index++];

  /* 次の空白/終端文字が現れるまでバッファに文字を読む */
  ch = line_buffer[line_index++];
  while (ch != ' ' && ch != '\t' && ch != '\0') {
    ret += ch;
    ch = line_buffer[line_index++];
  }

  /* 行バッファの末尾（改行）に到達しているかチェック. */
  if ((ch = line_buffer[line_index++]) == '\0') {
    /* 到達していたらインデックスをリセットして, 再び呼び出す */
    line_index = -1;
    return next_word();
  }

  return ret;
}
    

  

    
