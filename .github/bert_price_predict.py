from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json


class PricePredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model_name = None
        self.bert_model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.high_value_keywords = ['애플펜슬','아이패드','노트북','폰','지갑','카드','갤럭시탭','에어팟','워치','카메라','맥북']

    def extract_text_features(self, texts):
        features = []
        high_difficulty_words = ['두고', '놓고', '잃어', '분실', '찾아', '남겨', '급해', '빨리']
        urgency_words = ['급함', '긴급', '빠름', '서둘러', '당장']

        for text in texts:
            text = str(text).lower()
            feature_dict = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'high_difficulty_count': sum(1 for word in high_difficulty_words if word in text),
                'urgency_count': sum(1 for word in urgency_words if word in text),
                'has_time_constraint': 1 if any(word in text for word in ['시간', '분', '늦어']) else 0,
                'exclamation_count': text.count('!') + text.count('ㅠ') + text.count('ㅜ'),
                'has_high_value_item': 1 if any(word in text for word in self.high_value_keywords) else 0
            }
            features.append(feature_dict)
        return pd.DataFrame(features)

    def preprocess_data(self, df):
        df['text'] = df['요청글 제목'].fillna('') + ' ' + df['요청글 내용'].fillna('')
        text_features = self.extract_text_features(df['text'])

        bert_embeddings = self.bert_model.encode(df['text'].tolist(), convert_to_numpy=True)
        bert_df = pd.DataFrame(bert_embeddings, columns=[f'bert_{i}' for i in range(bert_embeddings.shape[1])])

        df['hour_sin'] = np.sin(2 * np.pi * df['요청 시각'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['요청 시각'] / 24)
        df['is_night'] = ((df['요청 시각'] >= 22) | (df['요청 시각'] <= 6)).astype(int)
        df['is_rush_hour'] = ((df['요청 시각'].between(7, 9)) | (df['요청 시각'].between(17, 19))).astype(int)

        categorical_cols = ['날씨', '주말여부']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])

        feature_cols = ['hour_sin', 'hour_cos', 'is_night', 'is_rush_hour'] + \
                       [f'{col}_encoded' for col in categorical_cols if col in df.columns]

        numerical_features = df[feature_cols]
        combined_features = pd.concat([text_features, bert_df, numerical_features], axis=1)
        self.feature_columns = combined_features.columns.tolist()  # 컬럼 이름 저장
        return combined_features, df['최종가격']

    def train_models(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models_config = {
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'param_dist': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [15, 31, 50],
                    'min_child_samples': [5, 10, 20],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'param_dist': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            }
        }

        results = {}
        best_r2 = -np.inf

        for name, cfg in models_config.items():
            print(f"\n=== {name.upper()} 모델 RandomizedSearchCV 훈련 ===")
            random_search = RandomizedSearchCV(
                estimator=cfg['model'],
                param_distributions=cfg['param_dist'],
                n_iter=10,
                scoring='r2',
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                'model': best_model,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"최적 파라미터: {random_search.best_params_}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = best_model.feature_importances_

            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name

        self.models = {name: result['model'] for name, result in results.items()}
        print(f"\n✅ Best model: {self.best_model_name} (R² = {best_r2:.4f})")
        return results

    def save_model(self, filepath='price_model'):
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, f'{filepath}.pkl')

        config = {
            'high_difficulty_words': ['두고', '놓고', '잃어', '분실', '찾아', '남겨', '급해', '빨리'],
            'urgency_words': ['급함', '긴급', '빠름', '서둘러', '당장'],
            'high_value_keywords': self.high_value_keywords,
            'best_model_name': self.best_model_name
        }

        with open(f'{filepath}_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)
        print(f"✅ 모델이 {filepath}.pkl에 저장되었습니다. Best 모델: {self.best_model_name}")


# ----------------------------
# 경량 예측기 클래스
# ----------------------------

class LightweightPricePredictor:
    def __init__(self, model_path='price_model.pkl', config_path='price_model_config.json'):
        self.model_data = joblib.load(model_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        best_model_name = self.config.get('best_model_name', list(self.model_data['models'].keys())[0])
        self.model = self.model_data['models'][best_model_name]
        self.scaler = self.model_data['scaler']
        self.label_encoders = self.model_data['label_encoders']
        self.bert_model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
        self.high_value_keywords = self.config['high_value_keywords']
        self.feature_columns = self.model_data.get('feature_columns', None)

    def extract_features(self, text, hour, weather, is_weekend):
        text = str(text).lower()
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'high_difficulty_count': sum(1 for w in ['두고','놓고','잃어','분실','찾아','남겨','급해','빨리'] if w in text),
            'urgency_count': sum(1 for w in ['급함','긴급','빠름','서둘러','당장'] if w in text),
            'has_time_constraint': 1 if any(w in text for w in ['시간','분','늦어']) else 0,
            'exclamation_count': text.count('!') + text.count('ㅠ') + text.count('ㅜ'),
            'has_high_value_item': 1 if any(w in text for w in self.high_value_keywords) else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'is_night': 1 if hour >= 22 or hour <= 6 else 0,
            'is_rush_hour': 1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0
        }

        if '날씨' in self.label_encoders:
            features['날씨_encoded'] = self.label_encoders['날씨'].transform([weather])[0] if weather in self.label_encoders['날씨'].classes_ else -1
        if '주말여부' in self.label_encoders:
            features['주말여부_encoded'] = self.label_encoders['주말여부'].transform([is_weekend])[0] if is_weekend in self.label_encoders['주말여부'].classes_ else -1

        bert_embedding = self.bert_model.encode([text])[0]

        feature_values = list(features.values()) + list(bert_embedding)
        if self.feature_columns is not None:
            return pd.DataFrame([feature_values], columns=self.feature_columns)
        else:
            return pd.DataFrame([feature_values])

    def predict(self, text, hour, weather, is_weekend):
        try:
            X = self.extract_features(text, hour, weather, is_weekend)
            X_scaled = self.scaler.transform(X)
            price = self.model.predict(X_scaled)[0]
            return max(100, round(price / 100) * 100)
        except Exception as e:
            print("예측 중 오류:", e)
            return 1000


if __name__ == "__main__":
    df = pd.read_excel("generated_training_data.xlsx", engine="openpyxl")

    predictor = PricePredictionModel()
    X, y = predictor.preprocess_data(df)
    results = predictor.train_models(X, y)

    for name, result in results.items():
        print(f"\n모델: {name}")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  MAE: {result['mae']:.2f}")
        print(f"  RMSE: {result['rmse']:.2f}")

    predictor.save_model()

    lightweight_predictor = LightweightPricePredictor()

    test_cases = [
        ("아 IT5호관에 애플펜슬 놔두고 옴 혹시 로비에 아직 있나여..ㅠㅠ", 14, "맑음", "평일"),
        ("중앙 도서관 3층 5번 구역 책상 위에 지갑 까먹고 놔두고 왔는데 거기 있나요 제발 급해요", 20, "비", "주말"),
        ("핸드폰 충전기 좀 빌려주세요", 10, "맑음", "평일"),
        ("아 오도 16번자리에 충전기 놔두고 왔는데 혹시 거기 있나요 제발",1,"비","주말"),
        ("지금 센파에 사람 많음?",20,"맑음","평일"),
        ("신관 2층 16번 책상에 아이패드 검은색 있나요.. 놔두고 왔는데",10,"맑음","평일"),
        ("공식당 세븐일레븐에 프로틴있음?",16,"맑음","주말"),
        ("아 에어팟 잃어버렸는데 중도 2층 책상에 있음?",22,"눈","평일")
    ]

    for i, (text, hour, weather, weekend) in enumerate(test_cases, 1):
        price = lightweight_predictor.predict(text, hour, weather, weekend)
        print(f"\n테스트 {i}: {price}원")
        print(f"  - 텍스트: {text}")
