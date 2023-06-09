公開されている通りに yml ファイルから環境構築すると，不明なパッケージが見つかり，エラーが吐かれる.
エラーが出たパッケージを削除していけば OK.

削除後の yml ファイルがこのディレクトリの environment.yml.
また, clip 等のパッケージがここには含まれていないので, 追加パッケージは requirements.txt に記す.
なお, clip は pypl からではなく, clip の github からダウンロードすること.

```
conda update conda
conda env create -f environment.yml
conda activate tta
pip install -r requirements.txt
```
