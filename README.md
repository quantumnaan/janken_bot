# これは何
じゃんけんのwebアプリ．
flask, flask-socketioでデータをやり取りしている．

# ファイルベースの説明

- src/main_app.py:
  リアルタイムでのブラウザとの通信，手の選択，データの保存を行う
- src/bayes_estimation.py: 
  pca.py で推定された事前分布のパラメータを用いてブラウザ側から送信されたリアルタイムのじゃんけんデータから相手の手の出し方をベイズ推定する
- src/em_estimation.py:
  事前分布として何を使うかによっては使用する

- document/janken_bot.pdf:
  このbotの構造を決めるにあたって考えたことを書いてます