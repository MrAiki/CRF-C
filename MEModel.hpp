#ifndef MEMODEL_H_INCLUDED
#define MEMODEL_H_INCLUDED

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <stdlib.h>

#include "MEFeature.hpp"

/* 学習繰り返し回数・収束判定定数のデフォルト値 */
static const int    MAX_ITERATION_LEARN = 5000;
static const double EPSILON_LEARN       = 10e-3;
static const int    MAX_F_SIZE          = 5000;
static const double EPSILON_F_SELECTION = 10e-3;
static const int    MAX_ITERATION_FGAIN = 1000;
static const double EPSILON_FGAIN       = 10e-3;
static const int    MAX_CANDIDATE_F_SIZE = 10000; /* 学習データから得られる候補素性の最大数 */

/* Maximum Entropy Model（最大エントロピーモデル）のモデルを表現するクラス */
class MEModel {
private:
  std::ifstream          input_file;             /* 現在読み込み中のファイル */
  std::string            line_buffer;            /* 読み込みファイルの行文字列バッファ */
  int                    line_index;             /* 読み込みファイルの行文字列のインデックス. -1は改行時 */
  int                    maxN_gram;              /* 最大Nグラムのサイズ */
  std::vector<MEFeature> features;               /* モデルを構成する素性 */
  std::vector<MEFeature> candidate_features;     /* 学習データから得られた素性候補 */
  std::map<std::vector<int>, double> joint_prob;            /* 結合確率分布P(x,y)を表す配列. パターンはyを末尾にする. */
  std::map<std::vector<int>, double> cond_prob;             /* 条件付き確率分布P(y|x)を表す配列. こちらもパターンはyを末尾にする. */
  /* 経験確率は素性から入手する */
  std::map<std::string, int>  word_map;          /* 単語と整数の対応をとる連想配列（ハッシュ） */
  int                    unique_word_no;         /* ユニークな単語の数. */
  std::map<std::vector<int>, double>  norm_factor;          /* 正規化項Z(x). パターンを突っ込むと正規化項の値が得られるマップ */
  std::set<int>          setY_marginal;          /* 周辺素性を活性化させるyの集合Ym */
  std::map<std::vector<int>, std::set<int> > setY_cond; /* 条件付き素性を活性化させるyの集合Y(x) */
  int                    pattern_count;          /* 学習データに表れたパターン総数 */
  int                    pattern_count_bias;     /* 1素性パターンのカウント閾値（これ以下の素性パターンは切り捨て） */
  double                 epsilon_learn;          /* 学習収束判定用の小さな値 */
  int                    max_iteration_learn;    /* 学習の最大繰り返し回数 */
  double                 epsilon_f_select;            /* 素性選択の尤度収束判定用の小さな値 */
  int                    max_iteration_f_select;      /* 素性選択の最大繰り返し回数（素性の最大数）*/
  double                 epsilon_f_gain;       /* 素性選択のゲイン収束判定用の小さな値 */
  int                    max_iteration_f_gain; /* 素性選択のゲイン取得用の最大繰り返し回数 */
  double                 max_sum_feature_weight; /* 最大の素性重み和C */

public:   
  /* コンストラクタ. maxN_gram以外はデフォルト値を付けておきたい */
  MEModel(int maxN_gram, int pattern_count_bias,
	  int max_iteration_l=MAX_ITERATION_LEARN, double epsilon_l=EPSILON_LEARN,
	  int max_iteration_f=MAX_F_SIZE, double epsilon_f=EPSILON_F_SELECTION,
	  int max_iteration_fgain=MAX_ITERATION_FGAIN, double epsilon_fgain=EPSILON_FGAIN);
  /* デストラクタ. 分布と正規化項の解放 */
  ~MEModel(void);

  /* 以下, メソッド */
public:
  /* ファイル名の配列を受け取り, 一気に読み込ませる. 経験確率/経験期待値をセット/更新する */
  void read_file_str_list(std::vector<std::string> filenames);
  /* モデルの確率分布の計算. 正規化項と素性の期待値の計算も同時に行う. */
  void calc_model_prob(void);
  /* 拡張反復スケーリング法で素性パラメタの学習を行う */
  void learning(void);
  /* 素性選択を行う */
  void feature_selection(void);
  /* 引数の文字列パターンの条件付き確率P(y|x)を計算する */
  double get_cond_prob(std::string *pattern_str_x, std::string pattern_str_y);
  /* 引数のxの文字列パターンから, 最も確率の高いyを予測として返す */
  std::string predict_y(std::vector<std::string> pattern_str_x);
  /* 上位ranking_sizeの確率のyを引数のrankingにセットする */
  void set_ranking(std::vector<std::string> pattern_str_x, std::vector<std::string> ranking);
  /* 候補素性情報の印字 */
  void print_candidate_features_info(void);
  /* モデル素性情報の印字 */
  void print_model_features_info(void);
 
private:
  /* 現在読み込み中のファイルから次の単語を返すサブルーチン */
  std::string next_word(void);
  /* ファイルから単語列を読み取り, 素性候補, 素性カウント, 単語マップを更新する. */
  void read_file(std::string filename);
  /* 経験確率と経験期待値を素性にセット/更新する */
  void set_empirical_prob_E(void);
  /* 周辺素性フラグのセット/更新 */
  void set_marginal_flag(void);
  /* 周辺素性を活性化させる要素の集合Ym, 条件付き素性を活性化させる集合Y(x)のセット */
  void set_setY(void);
  /* 引数のパターンでの, 全てのモデル素性の(パラメタ*重み)和を計算して返す */
  double get_sum_param_weight(std::vector<int> test_x, int test_y);
  /* 正規化項を計算してmapに結果をセットする */
  void calc_normalized_factor(void);
  /* ゲイン計算で用いるQ(feature^(pow)|pattern_x)を計算するサブルーチン */
  double calc_alpha_cond_E(int pow, MEFeature feature, int xlength, int *pattern_x, double alpha);
  /* ゲイン計算で用いる正規化項を計算するサブルーチン */
  double calc_alpha_norm_factor(MEFeature feature, int xlength, int *pattern_x, double alpha);
  /* ゲイン計算で用いる素性追加時の素性の期待値を計算するサブルーチン */
  double calc_alpha_E(MEFeature feature, double alpha);
  /* (テスト用)候補素性をモデル素性にコピーする */
  void copy_candidate_features_to_model_features(void);
  
};

#endif /* MEMODEL_H_INCLUDED */
