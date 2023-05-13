- Image/Text encoder 導入
- Student, Teacher をを Propmt Net に切り替え
- Prompt Net の初期化をどうするか？
- Doubly Stocastic Matrix の co-clustering
- Loss をどう設計するか？
  Sub Cluster 同士の相互情報量を大きくするのがベスト？
  Sub Cluster 内での variance は小さくしておきたい．
  そもそもドメインごとにまとまらせるのはどうやる？
  → 類似度行列そのままを co-clustering してもクラス特徴でまとまるよな．
  Prompt Distribution Learning の時，Prompt に ClassName を付加せずに Prompt Template(?)のみを入力して，Prompt Collection の多様性を大きくしていた．
  今回も，ClassName は付加させず（つまりはドメイン情報のみとなるか？）のみの Prompt が有効活用できそう．
  学部時代の研究は，クラス情報を壊し，Random Pseudo Label を振って Network にドメイン特徴量を学習させるというものだった．
  Google ＋松尾研究室の論文から，Class Vector に Domain-Specific Vector を付加した Prompt を用いると DG 性能が上がるというものがあった．
  よって，本研究では，画像のクラス情報を破壊して CLIP の Image Encoder にかけ，Domain Encoder に入力し，そのまま Class Name を付加しないで類似度行列を作成し，
  二重確率行列に直して co-clustering を行うことで，Domain Loss を計算し，Domain-Specific Feature を取得する Network を Student と Teacher で構築すれば良いのでは？
  ClassName の代わりに RPL を付加しても良いかもしれない．
  こうして，Prompt は Domain-specific な特徴を持つようになり，似たドメインで纏まりつつ，同じクラスでまとまるようになる
