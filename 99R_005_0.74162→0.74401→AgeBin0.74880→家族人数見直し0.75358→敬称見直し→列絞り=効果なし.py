##### 年齢の欠損値補完4(RandomForestRegressorで予測)

import pandas as pd
import numpy as np

##### データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

##### 前準備
all_data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

##### 家族人数のグループ分け
def family_size_group(size):
    if size == 1:   return 'Alone'
    elif size <= 4: return 'Small'
    elif size <= 7: return 'Middle'
    else:           return 'Large'

##### 特徴量エンジニアリング(家族人数)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['FamilySizeGroup'] = all_data['FamilySize'].apply(family_size_group)

##### 特徴量エンジニアリング(敬称抽出)
all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms',  'Lady'], 'Miss')
all_data['Title'] = all_data['Title'].replace(['Lady'],'Miss')
all_data['Title'] = all_data['Title'].replace(['Mme', 'Countess'], 'Mrs')
all_data['Title'] = all_data['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'],'Mr')

##### 特徴量エンジニアリング(Sexを数値化)
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})

##### 特徴量エンジニアリング(Fareの欠損値補完)
# これはTestデータに1件だけなので中央値で。特にこだわりなし
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

##################### AgeをRandomForestRegressorで推定 ここから
##### 推定に使用する項目を指定
age_pred_data = all_data[['Age', 'Pclass', 'Sex', 'Fare', 'SibSp', 'Parch', 'Title', 'FamilySizeGroup']]

##### ラベル特徴量をOne-Hotエンコーディング
age_pred_data = pd.get_dummies(age_pred_data)

##### Ageがわかっているデータとわかってないデータに分離し、numpyに変換
age_known  = age_pred_data[age_pred_data['Age'].notnull()].values
age_unknown= age_pred_data[age_pred_data['Age'].isnull()].values

##### 学習用データをX_age, y_ageに分離
X_age = age_known[:, 1:] # Age以外の特徴量
y_age = age_known[:, 0]  # Age(目的変数)

##### ランダムフォレスト(回帰)で推定モデルを構築
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X_age, y_age)

##### 欠損値のAge予測実行
predicted_ages = rfr.predict(age_unknown[:, 1:])

##### 元のall_dataに補完
all_data.loc[all_data['Age'].isnull(), 'Age'] = predicted_ages
#####################AgeをRandomForestRegressorで推定 ここまで

###### 年齢のbin化(予測後のAgeに対して実行)
labels = ['Child', 'Teen', 'Adult', 'Mid', 'Senior']
bins = [0, 12, 18, 31, 60, 100]
all_data['AgeBin'] = pd.cut(all_data['Age'], bins=bins, labels=labels, right=False).astype(str)

##### 本番モデル用のOne-Hot Encoding
all_data = pd.get_dummies(all_data, columns=['Title', 'FamilySizeGroup', 'AgeBin'])

##### 元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()

##### 特徴量を選択
#features = ["Pclass", "Sex", "Fare", 'Age'] \
#         + [col for col in train_data.columns if "Title_" in col] \
#         + [col for col in train_data.columns if "FamilySizeGroup_" in col]
#features = ["Pclass", "Sex", "Fare"] \
#         + [col for col in train_data.columns if "Title_" in col] \
#         + [col for col in train_data.columns if "FamilySizeGroup_" in col] \
#         + [col for col in train_data.columns if "AgeBin_" in col]
features = [
    "Pclass", "Sex", "Fare",
    # 効いているTitleだけを個別に指定(Rev, Dr, Dona などを除外)
    "Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master",
    # FamilySizeGroupはLargeも含めてすべて採用(お互いに補完し合うため)
    "FamilySizeGroup_Small", "FamilySizeGroup_Middle", "FamilySizeGroup_Alone", "FamilySizeGroup_Large",
    # AgeBinもすべて採用(SeniorやTeen単体は低くても、全体で1つの年齢軸を作るため)
    "AgeBin_Child", "AgeBin_Adult", "AgeBin_Mid", "AgeBin_Teen", "AgeBin_Senior"
]

##### 学習データを準備
X      = train_data[features]
y      = train_data["Survived"]
X_test = test_data[features]

##### モデル作成・学習
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1)
model.fit(X, y)

##### 予測
predictions = model.predict(X_test).astype(int)

##### 特徴量の貢献度 ここから
import matplotlib.pyplot as plt
import seaborn as sns

##### 1. モデルから特徴量の重要度（貢献度）を取得
importances = model.feature_importances_

##### 2. 特徴量名と重要度をデータフレームにまとめる
df_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False) # 貢献度が高い順にソート

##### --- 可視化（棒グラフで表示） ---
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
plt.title('Feature Importances (Titanic Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

##### --- 数値データでもランキングを表示 ---
print("\n--- 特徴量重要度ランキング ---")
print(df_importance.to_string(index=False))
##### 特徴量の貢献度 ここまで

##### 別の特徴量の貢献度 ここから
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ランダムフォレストを構成する木の中から、最初の1本（index=0）を取り出す
estimator = model.estimators_[0]

# 視覚化（複雑になりすぎないよう、深さを3までに制限）
plt.figure(figsize=(20, 10))
plot_tree(estimator, 
          feature_names=features, 
          class_names=['Perished', 'Survived'], # 0:死亡, 1:生存
          max_depth=3, 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.show()
##### 別の特徴量の貢献度 ここまで

##### 提出ファイル作成
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": predictions
})

##### 提出ファイル出力
submission.to_csv("submission-99R_004_0.75358_001.csv", index=False)

##### 完了
print("submission-99R-004_0.75358_001.csv を作成しました")
