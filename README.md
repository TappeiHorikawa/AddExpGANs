# GANを追実験し能力を確認する

言われた通りにネットワークを構築して言われた通りの成果が得られるか確認する。
ブランチごとに異なるネットワークを構築し各効果を確認する。

## 作成済みネットワーク

- GAN(MNIST生成)
- DCGAN(MNIST生成)
- SGAN(MNISTクラス判定)

## 参考文献

実践GAN　敵対的生成ネットワークによる深層学習 (Compass Booksシリーズ) : <https://www.amazon.co.jp/gp/product/B08573Y8GP/ref=ppx_yo_dt_b_d_asin_title_o02?ie=UTF8&psc=1>
上記本のGithubリポジトリ : <https://github.com/GANs-in-Action/gans-in-action>

## 開発メモ

どうやら学修率の不平等さが安定学習のキモのようである。

ドロップアウト率は0.8にした所、最終的に識別器の正答率が悪くなり、GAN全体が悪化した。
0.5が良い感じのようである

カラーノイズのようなものができてしまっているため、周囲を滑らかにするようなCONVフィルタのパラメータが必要である。
