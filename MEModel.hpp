#ifndef MEMODEL_H_INCLUDED
#define MEMODEL_H_INCLUDED

#include <iostream>
#include <cmath>
#include <string>
#include <map>
#include <vector>

#include "MEFeature.hpp"

/* Maximum Entropy Model（最大エントロピーモデル）のモデルを表現するクラス */
class MEModel {
private:
  FILE                   *fp_buffer;             /* 現在読み込み中のファイルポインタ */
  int                    maxN_gram;              /* 最大Nグラムのサイズ */
  std::vector<MEFeature> features;               /* モデルを構成する素性 */
  std::vector<MEFeature> candidate_features;     /* 学習データから得られた素性候補 */
  double                 *joint_prob;            /* 結合確率分布P(x,y)を表す配列. 1次配列で何とかする. xの長さは(maxN_gram-1)で固定. */
  double                 *cond_prob;             /* 条件付き確率分布P(y|x)を表す配列. */
  std::map<std::string, int>  word_map;               /* 単語と整数の対応をとる連想配列（ハッシュ） */
  double                 **norm_factor;          /* 正規化項Z(x). maxN_gram個の1次配列で各次元で1つのパターン長さの正規化項を計算しておく. ex)norm_factor[0]は長さ0でノルム(ユニグラム), norm_factor[1]は長さ1で(単語数)^(1)の長さの一次配列, norm_factor[2]は長さ2 */
  int                    pattern_count;          /* 学習データに表れたパターン総数 */
  int                    pattern_count_bias;     /* 1素性パターンのカウント閾値（これ以下の素性パターンは切り捨て） */
  double                 epsilon_learn;          /* 学習収束判定用の小さな値 */
  int                    max_iteration_learn;    /* 学習の最大繰り返し回数 */
  double                 epsilon_f_select_gain;       /* 素性選択のゲイン収束判定用の小さな値 */
  int                    max_iteration_f_select_gain; /* 素性選択のゲイン取得用の最大繰り返し回数 */
  double                 epsilon_f_select;            /* 素性選択の尤度収束判定用の小さな値 */
  int                    max_iteration_f_select;      /* 素性選択の最大繰り返し回数（素性の最大数）*/
  double                 max_sum_feature_weight; /* 最大の素性重み和C */
   
  /* コンストラクタ. maxN_gram以外はデフォルト値を付けておきたい */
  MEModel(int maxN_gram, int pattern_count_bias, int max_iteration_l, double epsilon_l,
	  int max_iteration_f, double epsilon_f, int max_iteration_fgain, double epsilon_fgain);
  /* デストラクタ. 分布と正規化項の解放 */
  ~MEModel(void);

  /* 以下, メソッド */
public:
  /* ファイルから単語列を読み取り, 素性候補, 素性カウント, 単語マップ, 経験分布を作成/更新する */
  void read_file(FILE* fp);
  /* ファイル名の配列を受け取り, 一気に読み込ませる */
  void read_file_str_list(int size, std::string* filenames);
  /* 経験確率分布の計算. カウントバイアスを考慮し, 素性の経験期待値もセットする. (注:カウントのリセットはしないように) */
  void calc_empirical_prib(void);
  /* モデルの確率分布の計算. 正規化項と素性の期待値の計算も同時に行う. */
  void calc_model_prob(void);
  /* 拡張反復スケーリング法で素性パラメタの学習を行う */
  void learning(void);
  /* 素性選択を行う */
  void feature_selection(void);
  /* 引数の文字列パターンの条件付き確率P(y|x)を計算する */
  double get_cond_prob(std::string *pattern_str_x, std::string pattern_str_y);
  /* 引数のxの文字列パターンから, 最も確率の高いyを予測として返す */
  std::string predict_y(std::string *pattern_str_x);
  /* 上位ranking_sizeの確率のyを引数のrankingにセットする */
  void set_ranking(std::string *pattern_str_x, int ranking_size, std::string *ranking);

private:
  /* 現在読み込み中のファイルから次の単語を返すサブルーチン */
  std::string next_word(void);
  /* 正規化項を計算して配列に結果をセットする */
  void calc_normalized_factor(void);
  /* ゲイン計算で用いるQ(feature^(pow)|pattern_x)を計算するサブルーチン */
  double calc_alpha_cond_E(int pow, MEFeature feature, int xlength, int *pattern_x, double alpha);
  /* ゲイン計算で用いる正規化項を計算するサブルーチン */
  double calc_alpha_norm_factor(MEFeature feature, int xlength, int *pattern_x, double alpha);
  /* ゲイン計算で用いる素性追加時の素性の期待値を計算するサブルーチン */
  double calc_alpha_E(MEFeature feature, double alpha);
  
};

#endif /* MEMODEL_H_INCLUDED */
