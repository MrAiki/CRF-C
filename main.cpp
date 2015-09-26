#include "MEModel.hpp"
#include <cstdio>
#include <getopt.h>
#include <boost/filesystem.hpp>
#include <iostream>

static void print_usage(void);                                                 /* 使い方を印字 */
static std::vector<std::string> split(const std::string &str, char delim); /* 文字列をdelimで区切ってvectorにする */
static std::set<std::string> split_to_set(const std::string &str, char delim); /* 文字列をdelimで区切って集合にする */

int main(int argc, char **argv)
{
  int option;                                  /* 実行時引数で選ばれたオプション */
  int maxN_gram = 3;                           /* 最大の素性Nグラム数 */
  int count_bias = 1;                          /* カウントバイアス : 頻度がこの値以下の素性は削除される */
  std::vector<std::string> read_file_name_buf; /* 読み込むファイル名（フルパス）のバッファ */
  std::set<std::string>    extension_list;     /* 読み込む拡張子リスト */
  MEModel *model;                              /* 最大エントロピーモデル */

  namespace fs = boost::filesystem;            /* boostの名前空間 */

  /* オプション付きの引数の処理 */
  while ((option = getopt(argc, argv, "g:c:e:s")) != -1) {
    switch (option) {
      case 'g': /* 最大グラム数の指定 (デフォルト:3) */
        maxN_gram = strtol(optarg, (char **)NULL, 10);
        break;
      case 'c': /* カウントバイアス（指定した頻度以下の素性は削除）の指定 (デフォルト:1) */
        count_bias = strtol(optarg, (char **)NULL, 10);
        break;
      case 'e': /* 読み込むファイルの拡張子を指定 ex) -e ".c .h .hpp .cpp" */
        extension_list = split_to_set(std::string(optarg), ' ');
        break;
      case 's': /* 素性の保存 */
        break;
      case ':': /* 値が必要なオプションに値が設定されていない */ /* FALLTHRU */
        std::cout << "Error : may be forgotten option value" << std::endl;
      case '?': /* 無効なオプション */  /* FALLTHRU */
      default:
        print_usage();
        exit(1);
    }
  }

  std::cout << "N_gram : " << maxN_gram << " Bias : " << count_bias << std::endl;

  /* optindは引数インデックス */
  if (optind == argc) {
    print_usage();
    exit(1);
  }

  /* 読み込みファイルの走査 */
  for (; optind < argc; optind++) {
    fs::path path(argv[optind]);

    if (fs::is_regular_file(path)) {
      /* 単一ファイルの読み込み */
      read_file_name_buf.push_back(path.string());
      std::cout << "GET: " << path.string() << std::endl;
    } else if ( fs::is_directory(path) ) {
      /* ディレクトリの場合 : ディレクトリ以下を走査 */
      fs::recursive_directory_iterator last;
      for ( fs::recursive_directory_iterator itr(path); itr != last; itr++ ) {
        if (extension_list.count(itr->path().extension().string()) > 0) {
          std::cout << "GET: " << itr->path() << std::endl;
          read_file_name_buf.push_back(("./" + itr->path().string()));
        }
      }
    }
  }

  model = new MEModel(maxN_gram, count_bias);
  model->read_file_str_list(read_file_name_buf);
  model->print_candidate_features_info();
  model->feature_selection();
  model->print_model_features_info();

  /* REPL(インタラクティブ)に使いたい... */
  std::string repl_line;
  /* quitで終了も... うーん */
  while (1) {
    std::vector<std::string> pattern;
    std::cout << std::endl;
    std::cout << ">> ";
    std::cin >> repl_line;
    if (repl_line == "quit") {
      break;
    } else {
      pattern = split(repl_line, ' ');
      model->get_ranking(pattern, 10);
    }
  }

  return 0;

}

/* 引数の説明を印字 */
static void print_usage(void)
{
  std::cout << "Usage :" << std::endl;
  std::cout << "./mepredict [-g maxN_gram] [-c count_bias] [-s] [-l filename] [-e extensions] filedir" << std::endl;
  std::cout << "-g maxN_gram(int) : set maximum N-gram model length to maxN_gram" << std::endl;
  std::cout << "-c count_bias(int) : set count bias to count_bias." << std::endl;
  std::cout << "-s : save model features." << std::endl;
  std::cout << "-l : load model features from filename" << std::endl;
  std::cout << "-e : file extension list. ex) -e \".cpp .hpp .c .h\" " << std::endl;
  std::cout << "filedir : can directory name. If you set directory name, read all files are in the directory." << std::endl;
}

/* 文字列をdelimで区切ってvectorを返す */
static std::vector<std::string> split(const std::string &str, char delim){
  size_t current = 0, found;
  std::vector<std::string> ret;
  while((found = str.find_first_of(delim, current)) != std::string::npos){
    ret.push_back(std::string(str, current, found - current));
    current = found + 1;
  }
  ret.push_back(std::string(str, current, str.size() - current));
  return ret;
}

/* 文字列をdelimで区切ってsetを返す */
static std::set<std::string> split_to_set(const std::string &str, char delim){
  /* vectorからsetに変換するだけ */
  std::vector<std::string> split_str = split(str, delim);
  std::set<std::string> set_str(split_str.begin(), split_str.end());
  return set_str;
}
