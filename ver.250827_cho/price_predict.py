import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score , RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PricePredictionModel:
    def __init__(self):
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,4))
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.high_value_keywords = ['애플펜슬','아이패드','노트북','폰','지갑','카드','갤럭시탭','에어팟','워치','카메라','맥북']
        
    def extract_text_features(self, texts):
        """텍스트에서 키워드 기반 특성 추출"""
        features = []
        
        # 난이도 키워드
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
        """데이터 전처리"""
        # 텍스트 특성 추출
        df['text'] = df['요청글 제목'].fillna('') + ' ' + df['요청글 내용'].fillna('')
        text_features = self.extract_text_features(df['text'])
        
        # TF-IDF 특성
        tfidf_features = self.vectorizer.fit_transform(df['text']).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # 시간 특성
        df['hour_sin'] = np.sin(2 * np.pi * df['요청 시각'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['요청 시각'] / 24)
        df['is_night'] = ((df['요청 시각'] >= 22) | (df['요청 시각'] <= 6)).astype(int)
        df['is_rush_hour'] = ((df['요청 시각'].between(7, 9)) | (df['요청 시각'].between(17, 19))).astype(int)
        
        # 카테고리 인코딩
        categorical_cols = ['날씨', '주말여부']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # 특성 결합
        feature_cols = ['hour_sin', 'hour_cos', 'is_night', 'is_rush_hour'] + \
                      [f'{col}_encoded' for col in categorical_cols if col in df.columns]
        
        numerical_features = df[feature_cols]
        combined_features = pd.concat([text_features, tfidf_df, numerical_features], axis=1)
        
        return combined_features, df['최종가격']
    
    def train_models(self, X, y, max_search_time=600):
        """여러 모델 훈련 + RandomizedSearchCV"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_config = {
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'param_dist': {
                    'n_estimators': [50, 100, 200,300],
                    'max_depth': [3, 5, 7,9],
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
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            }
        }
        
        results = {}
        
        for name, cfg in models_config.items():
            print(f"\n=== {name.upper()} 모델 RandomizedSearchCV 훈련 ===")
            model = cfg['model']
            param_dist = cfg['param_dist']
            
            # RandomizedSearchCV 정의
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=30,       # 탐색 횟수
                scoring='r2',
                cv=3,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            
            # LightGBM / RandomForest는 스케일 필요 없음
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
            
            # Feature importance 저장
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = best_model.feature_importances_
        
        self.models = {name: result['model'] for name, result in results.items()}
        return results
    
    def predict_ensemble(self, X):
        """가중치 앙상블 예측"""
        # models 순서 고정: LightGBM, RandomForest
        lgb_model = self.models.get('lightgbm')
        rf_model = self.models.get('random_forest')
        
        # 각각 예측
        lgb_pred = lgb_model.predict(X)
        rf_pred = rf_model.predict(X)
        
        # 가중치 앙상블
        ensemble_pred = 0.8 * lgb_pred + 0.2 * rf_pred
        
        return ensemble_pred

    
    def save_model(self, filepath='price_model'):
        """모델 저장 (온디바이스용)"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        

        best_model_name = max(self.models.keys(), key=lambda name: self.models[name].score(X_test, y_test) if 'X_test' in locals() else 0)
        
        joblib.dump(model_data, f'{filepath}.pkl')
        

        config = {
            'high_difficulty_words': ['두고', '놓고', '잃어', '분실', '찾아', '남겨', '급해', '빨리'],
            'urgency_words': ['급함', '긴급', '빠름', '서둘러', '당장'],
            'high_value_keywords': ['애플펜슬','아이패드','노트북','폰','지갑','카드','갤럭시탭','에어팟','워치','카메라','맥북'],
            'best_model_name': best_model_name 
        }
        
        with open(f'{filepath}_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)
        
        print(f"✅ 모델이 {filepath}.pkl에 저장되었습니다.")
    
    def plot_feature_importance(self, model_name='lightgbm', top_n=15):
        """특성 중요도 시각화"""
        if model_name in self.feature_importance:
            importance = self.feature_importance[model_name]
            indices = np.argsort(importance)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'{model_name.upper()} - Top {top_n} Feature Importance')
            plt.bar(range(top_n), importance[indices])
            plt.xticks(range(top_n), [f'feature_{i}' for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()

class LightweightPricePredictor:
    def __init__(self, model_path='price_model.pkl', config_path='price_model_config.json'):
        """경량 예측기 초기화"""
        self.model_data = joblib.load(model_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 설정 파일에서 최고 성능 모델을 찾아 로드
        best_model_name = self.config.get('best_model_name', 'lightgbm')
        self.model = self.model_data['models'][best_model_name]
        
        self.vectorizer = self.model_data['vectorizer']
        self.scaler = self.model_data['scaler']
        self.label_encoders = self.model_data['label_encoders']
    
    def extract_text_features(self, text):

        text = str(text).lower()
        
        high_difficulty_words = self.config.get('high_difficulty_words', [])
        urgency_words = self.config.get('urgency_words', [])
        high_value_keywords = self.config.get('high_value_keywords', [])

        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'high_difficulty_count': sum(1 for word in high_difficulty_words if word in text),
            'urgency_count': sum(1 for word in urgency_words if word in text),
            'has_time_constraint': 1 if any(word in text for word in ['시간', '분', '늦어']) else 0,
            'exclamation_count': text.count('!') + text.count('ㅠ') + text.count('ㅜ'),
            'has_high_value_item': 1 if any(word in text for word in high_value_keywords) else 0
        }
    
    def predict(self, text, hour, weather, is_weekend):
        try:
     
            text_features = self.extract_text_features(text)
            tfidf_features = self.vectorizer.transform([text]).toarray()[0]

            # 시간 특성
            time_features = {
                'hour_sin': np.sin(2*np.pi*hour/24),
                'hour_cos': np.cos(2*np.pi*hour/24),
                'is_night': 1 if (hour>=22 or hour<=6) else 0,
                'is_rush_hour': 1 if (7<=hour<=9 or 17<=hour<=19) else 0
            }

            # 카테고리 특성
            categorical_features = {}
            for col, value in zip(['날씨', '주말여부'], [weather, is_weekend]):
                if col in self.label_encoders:
                    try:
                        categorical_features[f'{col}_encoded'] = self.label_encoders[col].transform([value])[0]
                    except ValueError:
                        categorical_features[f'{col}_encoded'] = -1 # unseen label
            
            feature_vector = list(text_features.values()) + \
                             list(tfidf_features) + \
                             list(time_features.values()) + \
                             list(categorical_features.values())
            
            X_input = np.array(feature_vector).reshape(1, -1)

            predicted_price = self.model.predict(X_input)[0]

            predicted_price = max(100, round(predicted_price/100)*100)

            return int(predicted_price)

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            return 1000 # 기본 값 1000원 반환


# 사용 예시
if __name__ == "__main__":


    print("### 가격 예측 모델 ###")
    
    df = pd.read_excel("generated_training_data.xlsx", engine="openpyxl")
    
    # 모델 훈련
    predictor = PricePredictionModel()
    X, y = predictor.preprocess_data(df)
    results = predictor.train_models(X, y)
    
    # 최고 성능 모델 출력
    best_model = max(results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"\n최고 성능 모델: {best_model[0].upper()}")
    print(f"CV R² Score: {best_model[1]['cv_mean']:.4f}")
    
    # 모델 저장
    predictor.save_model('price_model_v2')
    
    # 경량 예측기 테스트
    print("\n### 경량 예측기 테스트 ###")
    lightweight_predictor = LightweightPricePredictor('price_model_v2.pkl', 'price_model_v2_config.json')
    
    # 테스트 예측
    test_cases = [
        ("아 IT5호관에 애플펜슬 놔두고 옴 혹시 로비에 아직 있나여..ㅠㅠ", 14, "맑음", "평일"),
        ("중앙 도서관 3층 5번 구역 책상 위에 지갑 까먹고 놔두고 왔는데 거기 있나요 제발 급해요", 20, "비", "주말"),
        ("핸드폰 충전기 좀 빌려주세요", 10, "맑음", "평일"), 
        ("아 오도 16번자리에 충전기 놔두고 왔는데 혹시 거기 있나요 제발",1,"비","주말"),
        ("지금 센파에 사람 많음?",20,"맑음","평일"),
        ("신관 2층 16번 책상에 아이패드 검은색 있나요.. 놔두고 왔는데",10,"맑음","평일"),
        ("공식당 세븐일레븐에 프로틴있음?",16,"맑음","주말")

    ]
    
    for i, (text, hour, weather, weekend) in enumerate(test_cases, 1):
        price = lightweight_predictor.predict(text, hour, weather, weekend)
        print(f"테스트 {i}: {price}원")
        print(f"  - 텍스트: {text[:30]}...")
        print(f"  - 시간: {hour}시, 날씨: {weather}, 요일: {weekend}")
        print()
