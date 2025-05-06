
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import seaborn as sns
data = sns.load_dataset("titanic")
data.head()

# 예시 데이터 로딩
X = data[["age", "fare", "sex"]]
y = data["survived"]

# 수치형 / 범주형 구분
numeric_features = ["age", "fare"]
categorical_features = ["sex"]

# 전처리 정의
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")), # 평균값으로 결측값 채우기
    ("scaler", StandardScaler())   # 정규화(평균 0, 표준편차 1)
])

# 범주형 피처 전처리 정의
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # 성별(남/녀)을 0/1로 변환
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 전처리 적용
X_preprocessed = preprocessor.fit_transform(X)
print("Preprocessing completed.")
