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
}

/* デストラクタ. */
MEModel::~MEModel(void)
{
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
*/
void MEModel::read_file(std::string file_name)
{
  std::string str_buf;   /* 単語バッファ */
  std::vector<int> Ngram_buf(maxN_gram); /* 今と直前(maxN_gram-1)個の単語列. Ngram_buf[maxN_gram-1]が今の単語, Ngram_buf[0]が(maxN_gram-1)個前の単語 */
  std::vector<MEFeature>::iterator f_it; /* 素性のイテレータ */
  int file_top_count;                    /* ファイル先頭分の読み飛ばし. */

  /* ファイルのオープン */
  input_file.open(file_name.c_str(), std::ios::in);
  if ( !input_file ) {
    std::cerr << "Error : cannot open file \"" << file_name << "\"." << std::endl;
    return;
  }
  
  /* ファイルの終端まで単語を集める */
  file_top_count = 0;
  while ( (str_buf=next_word()) != "\EOF" ) {
    /* 今まで見たことがない（新しい）単語か判定 */
    if (word_map.count(str_buf) == 0) {
      /* 新しい単語ならば, マップに登録 */
      word_map[str_buf] = unique_word_no++;
      /* std::cout << "word \"" << str_buf << "\" assined to " << word_map[str_buf] << std::endl; */
    } 

    /* Ngram_bufの更新 */
    for (int n_gram=0; n_gram < (maxN_gram-1); n_gram++) {
      Ngram_buf[n_gram] = Ngram_buf[n_gram+1];
    }
    Ngram_buf[maxN_gram-1] = word_map[str_buf];

    /* 新しいパターンか判定.
       Ngram_bufの長さを変えながら見ていく. */
    for (int gram_len=0; gram_len < maxN_gram; gram_len++) {

      /* x, y のパターンをベクトルで取得 */
      int buf_y_pattern = Ngram_buf[maxN_gram-1];
      std::vector<int> buf_x_pattern(gram_len);
      for (int ii=0; ii < gram_len; ii++) {
        buf_x_pattern[ii] = Ngram_buf[(maxN_gram-1)-gram_len+ii];
      }

      /* ファイル先頭分の読み飛ばし. */
      if (file_top_count < gram_len) {
        file_top_count++;
        break;
      }

      /* 現在の素性集合を走査し,
         既に同じパターンの素性があるかチェック */
      for (f_it = candidate_features.begin();
          f_it != candidate_features.end();
          f_it++) {
        /* 完全一致で確かめる. 完全一致でないと短いグラムモデルにヒットしやすくなってしまう */
        if (f_it->strict_check_pattern(buf_x_pattern,
                                       buf_y_pattern) == true) {
          /* 同じパターンがあったならば, 頻度カウントを更新 */
          f_it->count++;
          break;
        }

      }

      /* 既出のパターンではなかった -> 新しく素性集合に追加 */
      if (f_it == candidate_features.end()
          && pattern_count < MAX_CANDIDATE_F_SIZE) {
        /* 新しい素性を追加 */
        candidate_features.push_back(MEFeature(gram_len+1,
                                               buf_x_pattern,
                                               buf_y_pattern));
        /* パターン数の増加 */
        pattern_count++;
      }

    }

  }

  /* ファイルのクローズ */
  input_file.close();

}

/* ファイル名の配列から学習データをセット.
   得られた素性リストに経験確率と経験期待値をセットする */
void MEModel::read_file_str_list(std::vector<std::string> filenames)
{

  std::vector<std::string>::iterator file_it;
  /* read_fileを全ファイルに適用 */
  for (file_it = filenames.begin();
       file_it != filenames.end();
       file_it++) {
    read_file(*file_it);
  }

  /* 経験確率と経験期待値をセット */
  set_empirical_prob_E();

  /* TODO:FIXME:以下はテスト用 */
  copy_candidate_features_to_model_features();
  calc_normalized_factor();

  std::cout << "There are " << pattern_count << " unique patterns." << std::endl;
}

/* 経験確率/経験期待値を素性にセットする */
void MEModel::set_empirical_prob_E(void)
{
  std::vector<MEFeature>::iterator f_it, ex_f_it;     /* 素性のイテレータ */
  int sum_count;                                      /* 出現した素性頻度総数 */
  int pattern_y;                                      /* 経験期待値を計算する素性のパターン */
  std::vector<int> pattern_x;

  /* 頻度総数のカウント. */
  /* TODO:バイアスpattern_count_biasの考慮 */
  sum_count = 0;
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {
    sum_count += f_it->count;
  }

  if (sum_count == 0) {
    std::cerr << "Error : total number of features frequency equal to 0. maybe all of feature's frequency smaller than bias" << std::endl;
    exit(1);
  }

  /* 経験確率のセット. 頻度を総数で割るだけ. */
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {
    f_it->empirical_prob
      = (double)(f_it->count) / sum_count;
    f_it->empirical_E = 0.0f;              /* 期待値のリセット */
  }

  /* 経験期待値のセット. 
     (注) 相互に参照して徐々に期待値を加算していくので, ループ内での期待値リセットは禁止 */
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {
    /* パターンの取得. これを全素性で得れば全パターンを網羅したことになる. */
    pattern_x = f_it->get_pattern_x();
    pattern_y = f_it->get_pattern_y();

    /* 計算の方法:一つのパターンについて,
       他の素性でも活性化していたらその重み*経験確率を足す */
    for (ex_f_it = candidate_features.begin();
        ex_f_it != candidate_features.end();
        ex_f_it++) {
      ex_f_it->empirical_E
        += ex_f_it->checkget_weight_emprob(pattern_x, pattern_y);
    }

  }

}

/* 周辺素性フラグのセット/更新 */
void MEModel::set_marginal_flag(void)
{
  std::vector<MEFeature>::iterator f_it, ex_f_it, exex_f_it; /* 素性のイテレータ */
  int pattern_y;                          /* 一つのパターン */
  std::vector<int> pattern_x, pattern_w;  /* 異なるxのパターン. */
  bool is_find;

  /* 計算法 : 一つのパターンに対して, 他のpattern_wを持ってきたときに, 素性が異なる値をとった時, 素性は条件付き素性 */
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {

    is_find = false;
    /* パターン生成時は候補素性から */
    for (ex_f_it = candidate_features.begin();
        ex_f_it != candidate_features.end();
        ex_f_it++) {
      /* パターンの取得 */
      pattern_x = ex_f_it->get_pattern_x();
      pattern_y = ex_f_it->get_pattern_y();

      for (exex_f_it = candidate_features.begin();
          exex_f_it != candidate_features.end();
          exex_f_it++) {
        /* 他のXパターンの取得 */
        pattern_w = exex_f_it->get_pattern_x();
        /* 条件付き素性判定 */
        if (f_it->checkget_weight(pattern_x, pattern_y)
            !=
            f_it->checkget_weight(pattern_w, pattern_y)) {
          f_it->is_marginal = false;
          is_find = true;
          break;
        }
	
      }

      /* 判定済みならばループを抜ける */
      if (is_find) break;
    }

    /* 反例が見つかってなければ, 周辺素性 */
    if (!is_find) f_it->is_marginal = true;
  }
    
}

/* 引数のパターンでの, 全てのモデル素性の(パラメタ*重み)和(=エネルギー関数値)を計算して返す */
double MEModel::get_sum_param_weight(std::vector<int> test_x, int test_y)
{
  std::vector<MEFeature>::iterator f_it;
  double sum;

  sum = 0;
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    sum += f_it->checkget_param_weight(test_x, test_y);
  }

  return sum;
}

/* 正規化項の計算 */
void MEModel::calc_normalized_factor(void)
{
  double marginal_factor;                /* 周辺素性の性質から計算できる項の値 */
  std::vector<MEFeature>::iterator f_it; /* 素性のイテレータ */
  /* まず周辺素性のフラグをセット */
  set_marginal_flag();
  /* Yの分割も行う */
  set_setY();


  /* 周辺素性の素性から計算できる分を計算 */
  marginal_factor = unique_word_no;      /* 集合Yの要素数は単語数と等しい */
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    if (!f_it->is_marginal)
      marginal_factor -= 1;
  }
}

/* 周辺素性を活性化させるyの集合Ymと, 
   条件付き素性を活性化させるyの集合Y(x)をセットする. */
void MEModel::set_setY(void)
{
  std::vector<MEFeature>::iterator f_it, exf_it;

  /* Ymのセット : Y（単語）の要素を走査し, 周辺素性が活性化される要素を集める */
  for (int word_y = 0; word_y < unique_word_no; word_y++) {
    for (f_it = features.begin();
        f_it != features.end();
        f_it++) {
      if (f_it->is_marginal 
          && word_y == f_it->get_pattern_y()) {
        /* 周辺素性を活性化させているYの要素（単語）を発見 
           -> 集合に追加. breakして次の単語のチェックに */
        setY_marginal.insert(word_y);
        std::cout << "Ym elm : " << word_y << std::endl;
        break;
      }
    }
  }

  /* Y(x)のセット : 一番外側でXの要素（パターン）を走査, その中で条件付き素性が活性化される要素を集める */
  for (f_it = candidate_features.begin();
      f_it != candidate_features.end();
      f_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      for (exf_it = features.begin();
          exf_it != features.end();
          exf_it++) {
        if (!exf_it->is_marginal
            && exf_it->check_pattern(f_it->get_pattern_x(),
              word_y)) {
          /* 条件付き素性かつ,
             その素性を活性化させる組み合わせを発見
             -> 集合に追加し, breakして次のYの要素（単語）へ */
          std::vector<int> tmp_ptn_x;
          for (int i = 0; i < f_it->get_N_gram()-1; i++) {
            tmp_ptn_x.push_back((f_it->get_pattern_x())[i]);
          }
          setY_cond[tmp_ptn_x].insert(word_y);
          std::cout << "Y(x) elm : " << word_y << std::endl;
          break;
        }
      }
    }
  }	  

}


/* 候補素性情報の印字 */
void MEModel::print_candidate_features_info(void)
{
  std::vector<MEFeature>::iterator f_it;     /* 素性のイテレータ */  

  std::cout << "******** Feature's info ********" << std::endl;
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {
    f_it->print_info();
    std::cout << "---------------------------------------" << std::endl;
  }
  std::cout << "There are " << candidate_features.size() << " candidate features." << std::endl;
}

/* モデル素性情報の印字 */
void MEModel::print_model_features_info(void)
{
  std::vector<MEFeature>::iterator f_it;     /* 素性のイテレータ */  

  std::cout << "******** Feature's info ********" << std::endl;
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    f_it->print_info();
    std::cout << "---------------------------------------" << std::endl;
  }
  std::cout << "There are " << features.size() << " model features." << std::endl;

}  

/* (テスト用;for debug)候補素性をモデル素性にコピーする */
void MEModel::copy_candidate_features_to_model_features(void)
{
  features.resize(candidate_features.size());
  std::copy(candidate_features.begin(), candidate_features.end(),
            features.begin());
}

