#include "MEModel.hpp"

/* コンストラクタ */
MEModel::MEModel(int maxN_gram, int pattern_count_bias,
                 int max_iteration_learn, double epsilon_learn,
		 int max_iteration_f_select, double epsilon_f_select,
		 int max_iteration_f_gain, double epsilon_f_gain)
{
  this->maxN_gram              = maxN_gram;
  this->pattern_count_bias     = pattern_count_bias;
  this->max_iteration_learn    = max_iteration_learn;
  this->epsilon_learn          = epsilon_learn;
  this->max_iteration_f_select = max_iteration_f_select;
  this->epsilon_f_select       = epsilon_f_select;
  this->max_iteration_f_gain   = max_iteration_f_gain;
  this->epsilon_f_gain         = epsilon_f_gain;
  unique_word_no               = 0;
  pattern_count                = 0;
  joint_prob                   = NULL;
  cond_prob                    = NULL;
}

/* デストラクタ. 分布と正規化項の解放 */
MEModel::~MEModel(void)
{
  if (joint_prob != NULL) {
    delete [] joint_prob; 
  }

  if (cond_prob != NULL) {
    delete [] cond_prob;
  }
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
      ret += "\EOF";
      return ret;     /* ファイルの終端に達している時はEOFだけからなる文字列を返す */
    }
  }
  
  /* とりあえず1文字目を読む */
  ch = line_buffer[line_index];

  /* 空白, タブ文字, その他区切り文字の読み飛ばし */
  while (ch == ' ' || ch == '\t')
    ch = line_buffer[++line_index];

  /* 行バッファの末尾（改行）に到達しているかチェック. */
  if (ch == '\0') {
    /* 到達していたらインデックスをリセットして, 再び呼び出す */
    line_index = -1;
    return next_word();
  }

  /* 次の空白/終端文字が現れるまでバッファに文字を読む */
  while (ch != ' ' && ch != '\t' && ch != '\0') {
    ret += ch;
    ch = line_buffer[++line_index];
  }

  return ret;
}

/* 引数文字列のテキストファイルをオープンし, 次を行う
   ・単語ハッシュの作成/更新
   ・素性候補の作成
   ・素性の頻度カウント
   ・経験確率分布のアロケート、作成/更新
*/
void MEModel::read_file(std::string file_name)
{
  std::string str_buf;   /* 単語バッファ */
  int *Ngram_buf = new int[maxN_gram]; /* 今と直前(maxN_gram-1)個の単語列. Ngram_buf[maxN_gram-1]が今の単語, Ngram_buf[0]が(maxN_gram-1)個前の単語 */
  std::vector<MEFeature>::iterator f_it; /* 素性のイテレータ */

  /* ファイルのオープン */
  input_file.open(file_name.c_str(), std::ios::in);
  if ( !input_file ) {
    std::cerr << "Error : cannot open file \"" << file_name << "\"." << std::endl;
    return;
  }
  
  /* ファイルの終端まで単語を集める */
  while ( (str_buf=next_word()) != "\EOF" ) {
    /* 今まで見たことがない（新しい）単語か判定 */
    if (word_map.count(str_buf) == 0) {
      /* 新しい単語ならば, マップに登録 */
      word_map[str_buf] = unique_word_no++;
      std::cout << "word \"" << str_buf << "\" assined to " << word_map[str_buf] << std::endl;
    } 

    /* Ngram_bufの更新 */
    for (int n_gram=0; n_gram < (maxN_gram-1); n_gram++) {
      Ngram_buf[n_gram] = Ngram_buf[n_gram+1];
    }
    Ngram_buf[maxN_gram-1] = word_map[str_buf];

    /* 新しいパターンか判定.
       Ngram_bufの長さを変えながら見ていく. */
    for (int gram_len=0; gram_len < maxN_gram; gram_len++) {
      /* 現在の素性集合を走査し,
	 既に同じパターンの素性があるかチェック */
      for (f_it = candidate_features.begin();
	   f_it != candidate_features.end();
	   f_it++) {
	if (f_it->check_pattern(gram_len+1,
				&(Ngram_buf[maxN_gram-gram_len-1]),
				Ngram_buf[maxN_gram-1]) 
	    == true) {
	  /* 同じパターンがあったならば, 頻度カウントを更新 */
	  f_it->count++;
	  break;
	}
      }

      /* 既出のパターンではなかった ->　新しく素性集合に追加 */
      if (f_it == candidate_features.end()
	  && pattern_count < MAX_CANDIDATE_F_SIZE) {
	/* 新しい素性を追加 */
	candidate_features.push_back(MEFeature(gram_len+1,
					       &(Ngram_buf[maxN_gram-gram_len-1]),
					       Ngram_buf[maxN_gram-1]));
	/* パターン数の増加 */
	pattern_count++;
      }
      
    }
    
  }

  std::cout << "There are " << pattern_count << " unique patterns." << std::endl;

  /* 解放... */
  delete [] Ngram_buf;

}
