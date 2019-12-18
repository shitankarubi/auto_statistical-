# 統計量表示
## 概要 
   データセット(csvファイル)を入力し,
   目的変数を入力すると
   データセットのサイズに応じて回帰分析を行い
   mse,coef,Interceptを表示する

## 内容
  現在はまだ試作段階で
  データ数が40<500であれば
  Lasso回帰,Elastic Net,リッジ回帰を行い
  上記の分析手法の順番で統計量(現段階ではmse,coef,Intercept)を表示(現段階は値だけ出て分析手法が出力していない)
  また、データ数が500を超えるとSGD(確率的勾配降下法)が行われるように設定している
  
## 課題点
  ・どの値がどの分析手法か出力
  ・欠損値処理
  ・分析手法を増やす
  ・統計量ごとの順位づけ
  ・出力結果保存
  ・クラス分け
  ・グラフ出力
  
