# ME-C
ME（最大エントロピーモデル）によるコード補完

学生の内になにかやっておきましょう企画(2)

構想
  - 最大エントロピーモデルを用いてコード補完をする.
  - 区切り文字を指定できるようにしたい。
  - 学習は...色んな所から収集？　若しくは自分のコードだけ？

感想
  - 思ったより良い性能ではなかった
  - 確率分布の表現が難しい. std::mapを使った...
  - 追加（補完）素性の挙動がきもい. できればなくしてしまいたい.
    * 追加素性の経験期待値の計算がこれで合っているのか不明
    * 追加素性のせいでY(x)がYと一致してしまう.(全てのYを活性化させる条件付き素性だから)

展望
  - CRFへの拡張(枝ポテンシャル追加)
