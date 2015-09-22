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
  std::vector<MEFeature>::iterator   f_it;

  /* read_fileを全ファイルに適用 */
  for (file_it = filenames.begin();
       file_it != filenames.end();
       file_it++) {
    read_file(*file_it);
  }

  /* pattern_count_bias, カウントバイアスの適用 
     規定の回数未満の頻度の素性は除外 */
  bool is_delete; /* 削除を行ったフラグ */
  /* 削除が発生しなくなるまでループを回す */
  do {
    is_delete = false;
    for (f_it = candidate_features.begin();
	 f_it != candidate_features.end();
	 f_it++) {
      if (f_it->count < pattern_count_bias) {
	/* バイアス以下の頻度の素性を削除. イテレータを破壊するので,
	   一旦ループを抜ける */
	candidate_features.erase(f_it);
	is_delete = true;
	break;
      }
    }
  } while (is_delete);

  /* パターン総数の確定 */
  pattern_count = candidate_features.size();

  std::cout << "There are " << pattern_count << " unique patterns." << std::endl;
  std::cout << "There are " << unique_word_no << " unique words" << std::endl;

  /* 素性削除後のパターンXの集合の作成, TODO:word_mapの縮小 */
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {
         /* 新しいXパターンの追加を試みる */
        std::vector<int> pattern_x = f_it->get_pattern_x();
        setX.insert(pattern_x);
  }

  /* 経験確率と経験期待値をセット */
  set_empirical_prob_E();

  /* TODO:FIXME:以下はテスト用 */
  copy_candidate_features_to_model_features();
  /* まず周辺素性のフラグをセット */
  set_marginal_flag();
  /* Yの分割も行う */
  set_setY();

  /* 追加素性の重みを計算 */
  calc_additive_features_weight();

}

/* 経験確率/経験期待値を素性にセットする */
void MEModel::set_empirical_prob_E(void)
{
  std::vector<MEFeature>::iterator f_it, exf_it;   /* 素性のイテレータ */
  std::set<std::vector<int> >      find_x_pattern; /* 計算済みのXのパターン */
  int sum_count;                                   /* 出現した素性頻度総数 */

  /* 頻度総数のカウント. */
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
    f_it->empirical_E = 0.0f;                       /* 期待値のリセット */
    empirical_x_prob[f_it->get_pattern_x()] = 0.0f; /* xの周辺経験分布P~(x)の初期化 */
  }

  /* 経験期待値/xの周辺経験分布のセット.
     注) 経験確率分布は候補素性のパターンのみで総和をとる（それで全確率） 
         真にXとYの組み合わせを試すと異なる結果になる事に注意 */
  for (f_it = candidate_features.begin();
       f_it != candidate_features.end();
       f_it++) {

    for (exf_it = candidate_features.begin();
	 exf_it != candidate_features.end();
	 exf_it++) {
      /* 経験期待値の計算 */
      f_it->empirical_E 
	+= f_it->checkget_weight_emprob(exf_it->get_pattern_x(),
					exf_it->get_pattern_y());
      
      /* 周辺経験分布P~(x)の計算:同じパターンxの経験確率を一度のみ足す. */
      if (exf_it->get_pattern_x() == f_it->get_pattern_x()) {
	/* Xのパターンでまだ発見されてなければ, 和を取る */
	if (find_x_pattern.count(exf_it->get_pattern_x()) == 0) {
	  empirical_x_prob[exf_it->get_pattern_x()] 
	    += exf_it->empirical_prob;
	}
      }
    }
    
    /* 発見したXのパターンの集合に追加 */
    find_x_pattern.insert(f_it->get_pattern_x());
  }

}

/* 周辺素性フラグのセット/更新 : ボトルネック... 
 * ヒューリスティクス:N-gramモデルを使う限り, 周辺素性はユニグラム素性に限る */
void MEModel::set_marginal_flag(void)
{
  std::vector<MEFeature>::iterator f_it;            /* 素性のイテレータ */
  // std::set<std::vector<int> >::iterator x_it, w_it; /* Xのパターンのイテレータ */
  // bool is_find;

  /* 計算法 : 一つのパターンx_it, word_yに対して, 他のw_itを持ってきたときに, 素性が異なる値をとった時, 素性は条件付き素性 */
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {

    /* ヒューリスティクスを使用 */
    if (f_it->get_N_gram() == 1) {
      f_it->is_marginal = true;
    }

    /* ナイーブな計算
    is_find = false;
    // 全てのパターンを走査 
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
        for (w_it = setX.begin(); w_it != setX.end(); w_it++) {
          // 条件付き素性判定
          if (f_it->checkget_weight(*w_it, word_y)
              != f_it->checkget_weight(*x_it, word_y)) {
            f_it->is_marginal = false;
            is_find = true;
            break;
          }
        }
        // 判定済みならば一番外側までループを抜ける(goto?) 
        if (is_find) break;
      }

      if (is_find) break;
    }
    // 反例が見つかってなければ, 周辺素性 
    if (!is_find) f_it->is_marginal = true;
  */
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
  std::set<int>::iterator y_m, y_x;           /* 周辺素性を活性化させる要素の集合Ym, 条件付き素性を活性化させる要素の集合Y(x)のイテレータ */
  std::set<std::vector<int> >::iterator x_it; /* Xのパターンのイテレータ */

  /* 結合確率の正規化項の計算... FIXME:嘘!! 全部のパターン和は不可能!! */
  /*
  joint_norm_factor = 0.0f;
  for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      double energy = 0.0f;
      for (f_it = features.begin();
	   f_it != features.end();
	   f_it++) {
	energy += f_it->checkget_param_weight(*x_it, word_y);
      }
      joint_norm_factor += exp(energy);
    }
  }
  */

  /* 周辺素性の素性から計算できる分marginal_factorを計算 */
  marginal_factor = unique_word_no - setY_marginal.size(); /* |Y-Ym| */
  /* Ymについての和 */
  for (y_m = setY_marginal.begin();
       y_m != setY_marginal.end();
       y_m++) {
    double energy_z_y = 0.0f; /* z(y)のexp内部の値 */
    for (f_it = features.begin();
	 f_it != features.end();
	 f_it++) {
      if (f_it->is_marginal &&
	  f_it->get_pattern_y() == *y_m) {
	/* 周辺素性, かつyのパターンが一致して活性化している素性の
	 　エネルギー関数値を加算 */
	energy_z_y += f_it->parameter * f_it->weight;
      }
    }
    /* z(y_m)の値を加算 */
    marginal_factor += exp(energy_z_y);
  }

  /* 以下, 各Z(x)を計算していく */
  for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
    /* 周辺素性による値で初期化 */
    norm_factor[*x_it] = marginal_factor;
    
    /* Y(x)についての和 */
    for (y_x = setY_cond[*x_it].begin();
	 y_x != setY_cond[*x_it].end();
	 y_x++) {
      double energy_z_y = 0.0f, energy_z_y_x = 0.0f; /* z(y), z(y|x)のエネルギー関数値 */
      for (f_it = features.begin();
	   f_it != features.end();
	   f_it++) {
	if (f_it->is_marginal
	    && f_it->get_pattern_y() == *y_x) {
	  energy_z_y += f_it->parameter * f_it->weight;
	} else if (!f_it->is_marginal) {
	  energy_z_y_x += f_it->checkget_param_weight(*x_it, *y_x);
	}
      }

      /* z(y|x) - z(y) の加算 */
      norm_factor[*x_it] += exp(energy_z_y) * ( exp(energy_z_y_x) - 1 );
    }
  }
  
}

/* モデルの確率分布・モデル期待値の素性へのセット */
void MEModel::calc_model_prob(void)
{
  std::vector<MEFeature>::iterator f_it;           /* 素性イテレータ */
  std::set<std::vector<int> >::iterator x_it;      /* 集合Xのイテレータ */

  /* まず, 正規化項Z(x)の計算 */
  calc_normalized_factor();
  
  /* モデルの確率分布の計算 */
  for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      /* 確率分布にセットするパターンの生成
	 xyの順にパターンを連結 */
      std::vector<int> pattern_xy = (*x_it);
      pattern_xy.push_back(word_y);
      /* エネルギー関数値の計算 */
      double energy = 0.0f;
      for (f_it = features.begin();
	   f_it != features.end();
	   f_it++) {
	energy += f_it->checkget_param_weight(*x_it, word_y);
      }
      /* 条件付き確率のセット */
      cond_prob[pattern_xy]  = exp(energy) / norm_factor[*x_it];
    }
  }

  /* 期待値のリセット */
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    f_it->model_E = 0.0f;
  }

  /* モデル期待値の素性へのセット  */
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    for (x_it = setX.begin();
	 x_it != setX.end();
	 x_it++) {
      /* yについての和 */
      double sum_y = 0.0f;
      for (int word_y = 0; word_y < unique_word_no; word_y++) {
	/* 確率分布から取り出すパターンの生成 */
	std::vector<int> pattern_xy = (*x_it);
	pattern_xy.push_back(word_y);
	sum_y += f_it->checkget_weight(*x_it, word_y) * cond_prob[pattern_xy];
      }
      /* yについての和に周辺経験分布を掛けて足していき, 近似 */
      f_it->model_E += sum_y * empirical_x_prob[*x_it];
    }
  }
         
}

/* 素性重みの総和を定数にする追加素性の重み計算 */
void MEModel::calc_additive_features_weight(void)
{
  double max_sum_xy = -DBL_MAX;                           /* 最大の素性重み和を与えるパターンの, 和の値.(定数C) */
  std::set<std::vector<int> >::iterator x_it;             /* xのパターンイテレータ */
  std::vector<MEFeature>::iterator      f_it;             /* 素性のイテレータ */
  std::map<std::vector<int>, double>::iterator add_f_it;  /* 追加素性のイテレータ */

  for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      /* 1つのパターンについての素性重み和をとる */
      double sum_xy = 0.0f;
      for (f_it = features.begin();
	   f_it != features.end();
	   f_it++) {
	sum_xy += f_it->checkget_weight(*x_it, word_y);
      }
      /* 最大値の更新 */
      if (sum_xy > max_sum_xy) {
	max_sum_xy = sum_xy;
      }
      /* 追加素性のパターン生成, 重み初期化 */
      std::vector<int> pattern_xy = (*x_it);
      pattern_xy.push_back(word_y);
      add_feature_weight[pattern_xy] = -sum_xy; /* (後でCを加算) */
    }
  }
      
  /* 定数Cのセット */
  max_sum_feature_weight = max_sum_xy;

  /* 追加素性の重み確定 : 全ての重みにCを足す */
  for (x_it = setX.begin(); x_it != setX.end(); x_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      std::vector<int> pattern_xy = *x_it;
      pattern_xy.push_back(word_y);
      add_feature_weight[pattern_xy] += max_sum_xy; 
    }
  }

}

/* 周辺素性を活性化させるyの集合Ymと, 
   条件付き素性を活性化させるyの集合Y(x)をセットする. */
void MEModel::set_setY(void)
{
  std::vector<MEFeature>::iterator f_it;
  std::set<std::vector<int> >::iterator x_it;

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
        break;
      }
    }
  }

  /* Y(x)のセット : 一番外側でXの要素（パターン）を走査, その中で条件付き素性が活性化される要素を集める */
  for (x_it = setX.begin();
       x_it != setX.end();
       x_it++) {
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      for (f_it = features.begin();
           f_it != features.end();
           f_it++) {
        if (!f_it->is_marginal
            && f_it->check_pattern(*x_it, word_y)) {
          /* 条件付き素性かつ,
             その素性を活性化させる組み合わせを発見
             -> 集合に追加し, breakして次のYの要素（単語）へ */
          setY_cond[*x_it].insert(word_y);
          break;
        }
      }
    }
  }	  

}

/* 学習:反復スケーリング法による */
void MEModel::learning(void)
{
  int    iteration_count = 0;                 /* 学習繰り返しカウント */
  double change_amount   = DBL_MAX;           /* 変化量=パラメタ変化のRMS（二乗平均平方根） */
  std::vector<MEFeature>::iterator f_it;      /* 素性イテレータ */
  std::vector<double> delta(features.size()); /* パラメタ変化量 */

  /* 学習ループ */
  calc_model_prob();
  while (iteration_count < max_iteration_learn
	 && change_amount > epsilon_learn) {
    change_amount = 0.0f;

    /* 変化量deltaの計算, 全体の変化量への加算 */
    for (int i = 0; i < (int)features.size(); i++) {
      delta[i]
	= log(features[i].empirical_E/features[i].model_E) / max_sum_feature_weight;
      change_amount += pow(delta[i],2);
    }

    /* モデルのパラメタを一斉に変更 */
    for (int i = 0; i < (int)features.size(); i++) {
      features[i].parameter += delta[i];
    }

    /* 確率分布/モデル期待値の再計算 */
    calc_model_prob();
    calc_likelihood();
    
    /* 全体の変化量, 学習繰り返しカウントの更新 */
    change_amount = sqrt(change_amount/delta.size());
    iteration_count++;
    std::cout << "[" << iteration_count << "] : " << "RMS Change Amount : " << change_amount << ", Likelihood : " << likelihood << std::endl; 
  }
} 

/* 内部表現のパターンから条件付き確率を得る. 未知のXパターンに対処 */
double MEModel::get_cond_prob(std::vector<int> pattern_x, int pattern_y)
{
  std::vector<int> pattern_xy = pattern_x;
  pattern_xy.push_back(pattern_y);

  if (norm_factor.count(pattern_x) == 1) {
    /* 既知のXパターンの時 */
    return cond_prob[pattern_xy];
  } else {
    /* 未知のXパターンの時:その場で確率値を計算 */
    double norm_factor_x = 0.0f; /* 分母Z(x) */
    double numerator     = 1.0f; /* 分子(=exp(0)) */
    std::vector<MEFeature>::iterator f_it;
    /* ユニグラムのみが活性化するエネルギー関数和がZ(x) */
    for (f_it = features.begin();
         f_it != features.end();
         f_it++) {
      if (f_it->get_N_gram() == 1) {
        norm_factor_x += exp(f_it->parameter * f_it->weight);
        /* パターンyでユニグラムが活性化すれば, 分子の値がそれのみで定まる */
        if (f_it->get_pattern_y() == pattern_y) {
          numerator = exp(f_it->parameter * f_it->weight);
        }
      }
    }

    /* (分子/Z(x))が確率値となる */
    return (numerator / norm_factor_x);

  }

}

/* 引数の文字列パターンの条件付き確率P(y|x)を計算して返す */
double MEModel::get_cond_prob_from_str(std::vector<std::string> pattern_x, std::string pattern_y)
{
  std::vector<int> coded_x;

  /* 文字列を内部表現（整数列）に直す */
  /* 文字列xパターンを, 最長maxN_gramのパターンに変換
     pattern_x.size() < maxN_gram-1 の場合は長さはpattern_x.size()に合わせる */
  for (int i = 0; i < (int)pattern_x.size(); i++) {
    if (i < (int)pattern_x.size()-(maxN_gram-1)) {
      continue;
    }
    coded_x.push_back(word_map[pattern_x[i]]);
  }

  /* 確率値を取得して返す */
  return get_cond_prob(coded_x, word_map[pattern_y]);
}
 
/* 引数のxの文字列パターンから, 最も確率の高い単語yを予測して返す */ 
std::string MEModel::predict_y(std::vector<std::string> pattern_x) 
{
  double              max_prob;               /* 最大の確率値 */
  int                 max_prob_index;         /* 最大の確率値を与えるインデックス */
  std::vector<int>    coded_x;             /* パターン */
  std::map<std::string, int>::iterator map_itr;         /* キーに対応する文字列を探索するイテレータ */

  /* pattern_xを内部表現に直す */
  for (int i = 0; i < (int)pattern_x.size(); i++) {
    if (i < (int)pattern_x.size()-(maxN_gram-1)) {
      continue;
    }
    coded_x.push_back(word_map[pattern_x[i]]);
  }
  
  /* 最大確率値を与えるyの探索 */
  max_prob = 0.0f;
  max_prob_index = -1;
  for (int word_y = 0; word_y < unique_word_no; word_y++) {
    /* 最大確率の更新 */
    if (get_cond_prob(coded_x, word_y) > max_prob) {
      max_prob       = get_cond_prob(coded_x, word_y);
      max_prob_index = word_y;
    }
  }

  /* 内部表現（整数）から文字列に変換して返す */
  for (map_itr = word_map.begin();
       map_itr != word_map.end();
       map_itr++) {
    if (map_itr->second == max_prob_index) {
      return map_itr->first;
    }
  }

  /* make compiler happy */
  return NULL;
}

/* 上位ranking_sizeの確率のyを返す */
std::vector<std::string> MEModel::set_ranking(std::vector<std::string> pattern_x, int ranking_size)
{
  /* ランキングのサイズ */
  int size = ranking_size;

  /* ランキングサイズが大きすぎる時は, 単語数に合わせる */
  if (ranking_size > unique_word_no) {
    std::cerr << "Warning : ranking size exceeds number of dataset unique words!" << std::endl;
    size = unique_word_no;
  }

  /* ランキング */
  std::vector<std::string>             ranking(ranking_size); /* ランキング */
  std::vector<double>                  prob_list(unique_word_no), sorted_prob_list(unique_word_no); /* 各単語の確率値, ソートした確率値リスト */
  std::vector<int>                     coded_x;            /* Xパターン */
  std::map<std::string, int>::iterator map_itr;               /* キーに対応する文字列を探索するイテレータ */
  std::vector<double>::iterator        rank_itr;              /* ランキング */

  /* pattern_xを内部表現に直す */
  for (int i = 0; i < (int)pattern_x.size(); i++) {
    if (i < (int)pattern_x.size()-(maxN_gram-1)) {
      continue;
    }
    coded_x.push_back(word_map[pattern_x[i]]);
  }

  /* 確率リストの作成 */
  for (int word_y = 0; word_y < unique_word_no; word_y++) {
    /* 確率の取得 */
    prob_list[word_y] 
      = sorted_prob_list[word_y] 
      = get_cond_prob(coded_x, word_y);
  }

  /* ソートした確率リストの生成 */
  std::sort(sorted_prob_list.begin(), sorted_prob_list.end(), 
	    std::greater<double>());

  /* ランキングの作成 */
  bool is_find;
  std::set<int> finded_y; /* 発見済みのy. (同じ確率だと, 何度も同じyが選ばれる) */
  for (int rank = 0; rank < size; rank++) {
    is_find = false;
    for (int word_y = 0; word_y < unique_word_no; word_y++) {
      /* rank番目の確率値を与える内部表現を見つけ,
	 見つけた内部表現から文字列を探し, ランキングにセット */
      if (fabs(sorted_prob_list[rank] - prob_list[word_y]) < DBL_EPSILON
	  && finded_y.count(word_y) == 0) {	
	finded_y.insert(word_y);
	for (map_itr = word_map.begin();
	     map_itr != word_map.end();
	     map_itr++) {
	  if (map_itr->second == word_y) {
	    ranking[rank] = map_itr->first;
	    is_find = true;
	  }
	}
      }
      if (is_find) break;
    }
    std::cout << "Rank " << rank+1 << " : " << ranking[rank];    
    std::cout << " Prob. : " << sorted_prob_list[rank] << std::endl;
  }

  return ranking;

}

/* 対数尤度（経験対数尤度）の計算とメンバへのセット */
void MEModel::calc_likelihood(void)
{
  double sum;
  std::vector<MEFeature>::iterator f_it;

  /* モデルの結合分布は厳密には計算出来ないので, 
     P(x,y) ~= P~(x)P(y|x)とする. */
  sum = 0.0f;
  for (f_it = features.begin();
       f_it != features.end();
       f_it++) {
    std::vector<int> pattern_xy = f_it->get_pattern_x();
    pattern_xy.push_back(f_it->get_pattern_y());
    
    sum += f_it->empirical_prob 
           * log( empirical_x_prob[f_it->get_pattern_x()] * cond_prob[pattern_xy]);
  }

  likelihood = sum;
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

