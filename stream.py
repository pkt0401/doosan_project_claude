import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from io import BytesIO

# 언어 설정 텍스트 정의 (간소화)
system_texts = {
    "Korean": {
        "title": "두산에너빌리티 AI 위험성 평가 시스템",
        "dataset_label": "데이터셋 선택",
        "activity_label": "작업활동 입력:",
        "analyze_btn": "분석 실행",
        "disclaimer": "두산에너빌리티 AI Risk Assessment는 국내 및 해외 건설현장 ‘수시위험성평가’ 및 ‘노동부 중대재해 사례’를 학습하여 개발된 자동 위험성평가 프로그램 입니다. 생성된 위험성평가는 반드시 수시 위험성평가 심의회를 통해 검증 후 사용하시기 바랍니다.",
        "result_template": (
            "{process} / {unit_work}을 분석 한 결과 {unit_work} 작업 중 {hazard}으로 인한 사고발생 위험이 {risk_grade}등급으로 나타났습니다.\n\n"
            "해당 단위작업 중 발생한 중대재해 사례는 총 {similar_cases}건으로 사고원인 및 방지대책을 사전에 숙지하여 동일한 중대재해가 발생하지 않도록 관리하여 주시기 바랍니다."
        )
    }
}

# 페이지 설정
st.set_page_config(
    page_title="AI 위험성 평가 시스템",
    layout="wide"
)

# 스타일 적용
st.markdown("""
<style>
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #4CAF50;
    }
    .download-btn {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None

# 데이터셋 옵션
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)"
}

# ------------ 유틸리티 함수 ------------

def load_data(selected_dataset_name):
    """데이터 로드 및 전처리 함수"""
    try:
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        df = df.rename(columns={
            '작업활동 및 내용\nWork & Contents': '작업활동',
            '유해위험요인 및 환경측면 영향\nHazard & Risk': '유해위험요인'
        })
        df['T'] = df['빈도'] * df['강도']
        df['등급'] = df['T'].apply(lambda x: determine_grade(x))
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return pd.DataFrame()

def determine_grade(value):
    """위험 등급 결정 함수"""
    if 16 <= value <= 25: return 'A'
    elif 10 <= value <= 15: return 'B'
    elif 5 <= value <= 9: return 'C'
    elif 3 <= value <= 4: return 'D'
    elif 1 <= value <= 2: return 'E'
    else: return '알 수 없음'

def embed_texts(texts, api_key):
    """텍스트 임베딩 생성 함수"""
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=[text], model="text-embedding-3-large")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings, dtype='float32')

def calculate_similar_cases(query_embedding, embeddings, threshold=0.7):
    """유사 사례 계산 함수"""
    similarities = np.dot(embeddings, query_embedding.T)
    return np.sum(similarities >= threshold)

def generate_hazard_prompt(activity, similar_cases):
    """유해위험요인 생성 프롬프트"""
    examples = "\n".join([f"작업활동: {c['작업활동']}\n유해위험요인: {c['유해위험요인']}" for c in similar_cases])
    return f"""
    다음은 유사한 작업활동 사례입니다:
    {examples}
    
    다음 작업활동에 대한 유해위험요인을 예측하세요:
    작업활동: {activity}
    유해위험요인:"""

def generate_improvement_prompt(hazard, freq, intensity):
    """개선대책 생성 프롬프트"""
    return f"""
    유해위험요인: {hazard}
    원래 빈도: {freq}, 원래 강도: {intensity}
    위험 등급: {determine_grade(freq*intensity)}
    다음을 포함하는 개선대책을 생성하세요:
    1. 구체적인 안전 조치
    2. 장비 개선 방안
    3. 교육 훈련 방안
    4. 모니터링 계획
    생성된 개선대책:"""

# ------------ 메인 인터페이스 ------------

st.title(system_texts["Korean"]["title"])

# API 키 입력
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

# 데이터셋 선택
selected_dataset = st.selectbox(
    system_texts["Korean"]["dataset_label"],
    options=list(dataset_options.keys())
)

# 작업활동 입력
activity = st.text_area(system_texts["Korean"]["activity_label"], height=100)

# 분석 실행 버튼
if st.button(system_texts["Korean"]["analyze_btn"]):
    if not api_key:
        st.warning("API 키를 입력해주세요.")
    elif not activity:
        st.warning("작업활동을 입력해주세요.")
    else:
        with st.spinner("분석 진행 중..."):
            # 데이터 로드
            df = load_data(dataset_options[selected_dataset])
            
            # 임베딩 생성 및 FAISS 인덱스 구축
            embeddings = embed_texts(df['작업활동'].tolist(), api_key)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            # 쿼리 임베딩 생성
            query_embedding = embed_texts([activity], api_key)[0]
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # 유사 사례 검색
            D, I = index.search(query_embedding.reshape(1, -1), 5)
            similar_cases = df.iloc[I[0].tolist()].to_dict('records')
            
            # 유해위험요인 생성
            hazard_prompt = generate_hazard_prompt(activity, similar_cases[:3])
            hazard = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": hazard_prompt}],
                temperature=0.2
            ).choices[0].message.content
            
            # 위험도 계산
            freq = int(np.mean([c['빈도'] for c in similar_cases[:3]]))
            intensity = int(np.mean([c['강도'] for c in similar_cases[:3]]))
            t_value = freq * intensity
            risk_grade = determine_grade(t_value)
            
            # 개선대책 생성
            improvement_prompt = generate_improvement_prompt(hazard, freq, intensity)
            improvement = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.3
            ).choices[0].message.content
            
            # 유사 사례 수 계산
            similar_count = calculate_similar_cases(query_embedding, embeddings)
            
            # 결과 표시
            st.markdown("### 분석 결과")
            result_text = system_texts["Korean"]["result_template"].format(
                process=selected_dataset,
                unit_work=activity,
                hazard=hazard,
                risk_grade=risk_grade,
                similar_cases=similar_count
            )
            st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
            
            # 개선대책 표시
            st.markdown("### 개선대책")
            st.markdown(improvement)
            
            # Excel 다운로드 준비
            output_df = pd.DataFrame({
                "공정": [selected_dataset],
                "단위작업": [activity],
                "유해위험요인": [hazard],
                "빈도": [freq],
                "강도": [intensity],
                "T값": [t_value],
                "위험등급": [risk_grade],
                "유사사례수": [similar_count],
                "개선대책": [improvement]
            })
            
            # Excel 파일 생성
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False)
            
            # 다운로드 버튼
            st.download_button(
                label="결과 다운로드 (Excel)",
                data=output.getvalue(),
                file_name="위험성평가_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-btn'
            )

# 안내 문구 표시
st.markdown(f'<div style="margin-top: 50px;">{system_texts["Korean"]["disclaimer"]}</div>', 
            unsafe_allow_html=True)
