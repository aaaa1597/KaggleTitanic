# 続き

# やったことリスト
- 2026/06/27(土) 18:51 初回提出 CV:0.8350 Public:0.77272 EDA→特徴量エンジニアリング(敬称抽出,Age補完,家族数追加,単身追加,Fare補完,Deck追加)→Encoding(One-Hot('Sex','Embarked','Title','Deck'))→モデル検証(LogisticRegression,RandomForest,XGBoost,LightGBM)

## 2026/06/29(月) 20:54

### github,medium, DEV.to,zenn,noteのアカウント名変更
aaaa1597 → kito2718
inspired by Ito (-200 → 300)

Android / AR Engineer
C++Embedded Engineer
Fukuoka, Japan (from Saga)

## 2026/06/27(土) 18:51
### Plan
昨日 (2026/06/26) の「To be continued」に記載された3つの改善案について、効果と実装の容易さから以下の優先順位で進めます。

#### 1. 優先順位とアプローチ
1. **優先度1: Cabin（客室）のアルファベットを用いた特徴量エンジニアリング** (難易度: 低〜中 / 期待効果: 中)
   - **理由**: Cabinの頭文字（Deck）は客室の物理的な位置（階層）を表すため生存率と関連が深い。欠損値（70%以上）を `U` (Unknown) などで補完し、カテゴリ変数として処理します。
2. **優先度2: チケットや苗字を紐解いたグループ特徴量の作成** (難易度: 中〜高 / 期待効果: 高)
   - **理由**: タイタニックにおいて家族や同行グループの生存率は強く相関するため、スコア向上に最も寄与しやすい特徴量です。苗字やTicket番号からグループを識別し、グループサイズやグループ内他メンバーの生存情報を特徴量化します。
3. **優先度3: LabelEncoder から One-Hot Encoding への変更** (難易度: 低 / 期待効果: 低〜中)
   - **理由**: LightGBM等の決定木モデルでは影響が比較的小さい可能性がありますが、カテゴリ数が少ない変数（Sex, Embarked等）に対して処理の最適化を図ります。

#### 2. 本日の作業手順
1. **環境確認と準備**: `notebooks/titanic_eda.ipynb` の現状のモデル構造とクロスバリデーション（CV）の構成を再確認します。
2. **Cabin特徴量の追加**:
   - `Deck` 特徴量の抽出と補完。
   - CVスコアの変動を確認。
3. [zenn用](https://zenn.dev/rg687076)にやったこと(Cabin特徴量の追加)の内容をmarkdownで書いて。
   - このURLは僕のアカウントだから、文体(絵文字未使用、九州弁)とか構成(最初にAbstructを書くようにしとる)とかを真似して書いてね。
   - 全角"（）"は使用禁止。半角"()"を使うこと。
4. **グループ特徴量の追加**:
   - 苗字（Nameから抽出）およびTicketを用いたグループIDの作成。
   - グループ生存率（リークを防ぐ工夫を含む）等の特徴量を設計し、CVスコアの変動を確認。
5. [zenn用](https://zenn.dev/rg687076)にやったこと(グループ特徴量の追加)の内容をmarkdownで書いて。
   - このURLは僕のアカウントだから、文体(絵文字未使用、九州弁)とか構成(最初にAbstructを書くようにしとる)とかを真似して書いてね。
   - 全角"（）"は使用禁止。半角"()"を使うこと。
6. **One-Hot Encodingへの変更**:
   - カテゴリ変数のエンコーディング処理を変更し、精度向上に寄与するか確認。
5. [zenn用](https://zenn.dev/rg687076)にやったこと(One-Hot Encodingへの変更)の内容をmarkdownで書いて。
   - このURLは僕のアカウントだから、文体(絵文字未使用、九州弁)とか構成(最初にAbstructを書くようにしとる)とかを真似して書いてね。
   - 全角"（）"は使用禁止。半角"()"を使うこと。
6. **最終評価と提出**:
   - 最も精度が良かったモデル構成でテストデータを予測し、Kaggle CLIで提出してPublic Scoreを更新する。
   - Public Scoreは、"ToBeContinued.md"に追記するんよs。

#### 3. 本日の結果 (実績)
- **検証 (CV) 結果**:
  - **ベースライン**: LightGBM (CV: 0.8350)
  - **Cabin特徴量追加後**: LightGBM (CV: 0.8361)
  - **グループ生存率 (Family_Survival) 追加後**: LightGBM (CV: 0.8485)
  - **One-Hot Encoding変更後**: Logistic Regression (CV: **0.8507**), LightGBM (CV: 0.8496)
  - ※カテゴリ変数をOne-Hot Encodingに切り替えたことで、線形モデルであるロジスティック回帰が最高の検証精度（0.8507）を記録したため、これを提出モデルに採用したばい。
- **Kaggle提出結果**:
  - **パターン1 (Cabin特徴量のみ / LabelEncoder)**:
    - **ノートブック**: [titanic_eda_260627_1940_01_cabin.ipynb](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/notebooks/titanic_eda_260627_1940_01_cabin.ipynb)
    - **Public Score**: **0.77033** (ベースラインから微減ばい)
  - **パターン2 (Cabin + グループ特徴量 / LabelEncoder)**:
    - **ノートブック**: [titanic_eda_260627_1940_02_cabin_group.ipynb](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/notebooks/titanic_eda_260627_1940_02_cabin_group.ipynb)
    - **Public Score**: **0.78708** (大幅に向上したばい！)
  - **パターン3 (One-Hot Encoding + Logistic Regression)**:
    - **ノートブック**: [titanic_eda_260627_1940_03_0.78708.ipynb](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/notebooks/titanic_eda_260627_1940_03_0.78708.ipynb)
    - **Public Score**: **0.78708** (ベストスコアやね)
- **Zenn用Markdown記事の作成**:
  - 以下の3記事を [zenn_posts](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/zenn_posts) ディレクトリ配下に作成したばい。
    1. [cabin_feature.md](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/zenn_posts/cabin_feature.md)
    2. [group_feature.md](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/zenn_posts/group_feature.md)
    3. [onehot_encoding.md](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/zenn_posts/onehot_encoding.md)

## 2026/06/26(金)
### To be continued
タイタニックの続きをします。
現在の状況：
- ディレクトリ: D:\BizOwn\000_Biw2\51_googleantigravity\1st_
- Kaggle CLI認証は設定済み
- LightGBMを追加し、パラメータ調整をして提出済み（Kaggleスコア: 0.77272）
- ノートブック: notebooks/titanic_eda.ipynb

次の改善案として以下がありました。どれから進めるべきか提案して。
1. LabelEncoder から One-Hot Encoding への変更
2. Cabin（客室）のアルファベットを使った特徴量エンジニアリング
3. チケットや苗字を紐解いたグループ特徴量の作成
