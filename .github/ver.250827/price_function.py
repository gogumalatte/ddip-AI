import pandas as pd
import numpy as np
import random

lose_keywords = ['두고', '놓고', '잃어', '분실', '찾아', '남겨']
high_value_keywords = ['애플펜슬','아이패드','노트북','폰','지갑','카드','갤럭시탭','에어팟','워치','카메라','맥북']
medium_value_items = ['충전기', '우산', '텀블러', '전공 책', '필통', '모자', '보조배터리', '마우스', '학생증', '목도리']

locations = [
    '중앙도서관 1층 열람실', '백호관(농대) 101호', 'IT3호관', '공대 12호관', '글로벌플라자 1층',
    '복지관 학생식당', '제1학생회관', '경상대학 건물', '사회과학대학 열람실', '백양로', '일청담 근처 벤치',
    '대강당', '첨성관(기숙사) 로비', '중도 신관','중도 구관','보람관','누리관','공대6호관','인문대','경상대','자연대','미융관',
    '미래융합관','조형관','사과대','법대','경상대','4합','간호대','테크노빌딩','정센','정보센터','GS25','세븐일레븐','공식당','교직원식당',
    '첨성 카페테리아','복지관 교직원 식당','공대 7호관','공대8호관','공대9호관','공대10호관','공대1호관','오도','IT1호관','IT2호관','IT5호관','융복합관','융복','IT4호관',
    '조은문(북문) 근처 카페'
]
colors = ['검은색', '흰색', '파란색', '실버', '스페이스 그레이', '남색']
find_verbs = ['보신 분 계신가요?', '찾아주시면 사례하겠습니다', '아직 있을까요?', '습득하신 분 연락주세요', '보이면 알려주세요 제발ㅠㅠ']

# 날씨 정보
weather_bonus = {'맑음': 0, '비': 500, '눈': 500, '추움': 300, '더움': 400, '태풍': 1000, '황사': 200}
weather_conditions = list(weather_bonus.keys())
weather_probabilities = [0.6, 0.1, 0.05, 0.1, 0.1, 0.02, 0.03]


def classify_difficulty(text):
    if any(word in text for word in lose_keywords):
        if any(word in text for word in high_value_keywords):
            return '상'
        else:
            return '중'
    else:
        return '하'

def generate_sentence(item_list, difficulty):
    """난이도에 맞는 문장을 생성하는 범용 함수"""
    item = random.choice(item_list)
    location = random.choice(locations)
    title = f"{item} 분실했습니다ㅠㅠ"
    if random.random() < 0.5:
        title = f"{location}에서 {item} 보신 분?"
    
    desc_list = [f"{location}에 {item}을 놓고 온 것 같아요.."]
    if random.random() > 0.4:
        desc_list.append(f"{random.choice(colors)} 색상인데,")
    desc_list.append(random.choice(find_verbs))
    
    content = ' '.join(desc_list)
    return {'요청글 제목': title, '요청글 내용': content, '난이도': difficulty}



# 1. 원본 데이터 읽기
file_path = '250825_price.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df['text'] = df['요청글 제목'].fillna('') + ' ' + df['요청글 내용'].fillna('')

# 2. 원본 데이터의 난이도 우선 분류
df['난이도'] = df['text'].apply(classify_difficulty)
print("[증강 전] 데이터 분포:")
print(df['난이도'].value_counts())


num_high_current = df['난이도'].value_counts().get('상', 0)
num_medium_current = df['난이도'].value_counts().get('중', 0)

num_high_to_generate = max(0, 300 - num_high_current)
num_medium_to_generate = max(0, 300 - num_medium_current)

augmented_list = []
if num_high_to_generate > 0:
    print(f"\n'상' 난이도 데이터 {num_high_to_generate}개를 증강합니다...")
    augmented_list.extend([generate_sentence(high_value_keywords, '상') for _ in range(num_high_to_generate)])

if num_medium_to_generate > 0:
    print(f"'중' 난이도 데이터 {num_medium_to_generate}개를 증강합니다...")
    augmented_list.extend([generate_sentence(medium_value_items, '중') for _ in range(num_medium_to_generate)])

if augmented_list:
    df_augmented = pd.DataFrame(augmented_list)
    df = pd.concat([df, df_augmented], ignore_index=True)
    print("데이터 병합 완료.")



def assign_price(diff):
    if diff == '하': return 500
    elif diff == '중': return 1500
    else: return 4000

df['기본가격'] = df['난이도'].apply(assign_price)

df['요청 시각'] = [random.randint(0, 23) for _ in range(len(df))]
def get_time_surcharge(hour):
    if 1 <= hour < 7: return 700
    elif 7 <= hour < 9: return 500
    elif 18 <= hour or hour < 1: return 300
    else: return 0
df['시간보상'] = df['요청 시각'].apply(get_time_surcharge)

df['날씨'] = random.choices(weather_conditions, weights=weather_probabilities, k=len(df))
df['날씨보상'] = df['날씨'].map(weather_bonus)

df['주말여부'] = [random.choice(['평일', '주말']) for _ in range(len(df))]
df['주말보상'] = df['주말여부'].apply(lambda x: 300 if x=='주말' else 0)

def get_variation(diff):
    if diff=='하': return random.randint(-100,100)
    else: return random.randint(-200,200)
df['변동보상'] = df['난이도'].apply(get_variation)

numeric_cols = ['기본가격', '시간보상', '날씨보상', '주말보상', '변동보상']
df['최종가격'] = (df[numeric_cols].sum(axis=1) / 100).round() * 100

# 4. 최종 결과 확인 및 저장
print("\n[증강 후] 최종 데이터 분포:")
print(df['난이도'].value_counts())

df.to_excel("generated_training_data.xlsx", index=False)
print("\n학습 데이터 생성 완료.")
