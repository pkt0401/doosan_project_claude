import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# ------------------ 언어 설정 텍스트 ------------------
system_texts = {
    "Korean": {
        "title": "AI 위험성 평가 시스템",
        "tab_overview": "시스템 개요",
        "tab_main": "위험 평가 및 개선",
        "overview_header": "LLM 기반 자동 위험 평가 시스템",
        "overview_text": """... 한국어 설명 텍스트 ...""",
        "process_title": "AI 위험 평가 프로세스",
        "process_steps": ["작업 입력", "AI 분석", "위험 예측", "등급 산정", "개선안 생성", "안전 조치"],
        "features_title": "시스템 특징",
        "phase1_features": """... 한국어 특징 설명 ...""",
        "phase2_features": """... 한국어 특징 설명 ...""",
        "api_key_label": "OpenAI API 키 입력:",
        "dataset_label": "데이터셋 선택",
        "load_data_btn": "데이터 로드",
        "activity_label": "작업 활동 입력:",
        "predict_btn": "분석 실행",
        "similar_header": "유사 사례",
        "risk_header": "위험 평가 결과",
        "improve_header": "개선 대책",
        "result_activity": "작업 활동:",
        "result_hazard": "위험 요소:",
        "result_freq": "빈도:",
        "result_intensity": "강도:",
        "result_t": "T 값:",
        "result_grade": "위험 등급:",
        "improve_before": "개선 전",
        "improve_after": "개선 후",
        "reduction_rate": "위험 감소율:",
        "error_api": "API 키를 입력해주세요",
        "error_data": "먼저 데이터를 로드해주세요",
        "error_activity": "작업 활동을 입력해주세요"
    },
    "English": {
        "title": "AI Risk Assessment System",
        "tab_overview": "System Overview",
        "tab_main": "Risk Assessment",
        "overview_header": "LLM-based Automated Risk Assessment",
        "overview_text": """... English description ...""",
        "process_title": "AI Risk Assessment Process",
        "process_steps": ["Input", "AI Analysis", "Risk Prediction", "Grading", "Improvement", "Safety"],
        "features_title": "System Features",
        "phase1_features": """... English features ...""",
        "phase2_features": """... English features ...""",
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_btn": "Load Data",
        "activity_label": "Work Activity:",
        "predict_btn": "Run Analysis",
        "similar_header": "Similar Cases",
        "risk_header": "Risk Assessment",
        "improve_header": "Improvement Plan",
        "result_activity": "Activity:",
        "result_hazard": "Hazard:",
        "result_freq": "Frequency:",
        "result_intensity": "Intensity:",
        "result_t": "T Value:",
        "result_grade": "Risk Grade:",
        "improve_before": "Before",
        "improve_after": "After",
        "reduction_rate": "Risk Reduction:",
        "error_api": "Please enter API key",
        "error_data": "Load data first",
        "error_activity": "Enter work activity"
    },
    "Chinese": {
        "title": "人工智能风险评估系统",
        "tab_overview": "系统概述",
        "tab_main": "风险评估",
        "overview_header": "基于LLM的自动风险评估",
        "overview_text": """... 中文描述 ...""",
        "process_title": "风险评估流程",
        "process_steps": ["输入", "分析", "预测", "评级", "改进", "安全"],
        "features_title": "系统特点",
        "phase1_features": """... 中文特点 ...""",
        "phase2_features": """... 中文特点 ...""",
        "api_key_label": "输入OpenAI API密钥:",
        "dataset_label": "选择数据集",
        "load_data_btn": "加载数据",
        "activity_label": "工作活动:",
        "predict_btn": "开始分析",
        "similar_header": "类似案例",
        "risk_header": "风险评估",
        "improve_header": "改进计划",
        "result_activity": "活动:",
        "result_hazard": "危险:",
        "result_freq": "频率:",
        "result_intensity": "强度:",
        "result_t": "T值:",
        "result_grade": "风险等级:",
        "improve_before": "改进前",
        "improve_after": "改进后",
        "reduction_rate": "风险降低率:",
        "error_api": "请输入API密钥",
        "error_data": "请先加载数据",
        "error_activity": "输入工作活动"
    }
}

# ------------------ 스타일 설정 ------------------
st.markdown("""
<style>
    .main-title { font-size: 2.5em; color: #1a73e8; text-align: center; padding: 20px; }
    .section-title { font-size: 1.8em; color: #0d47a1; margin: 15px 0; }
    .result-box { padding: 15px; background: #f8f9fa; border-radius: 10px; margin: 10px 0; }
    .similar-case { padding: 12px; background: #e8f5e9; border-radius: 8px; margin: 8px 0; }
    .metric-box { padding: 15px; background: #fff3e0; border-radius: 10px; }
    .warning { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------ 데이터 처리 함수 ------------------
def load_dataset(name):
    try:
        df = pd.read_excel(f"{name}.xlsx")
        df = df.rename(columns={
            '개선대책': 'Improvement', 'Improvement Measures': 'Improvement', '改进措施': 'Improvement'
        })
        df['T'] = df['Frequency'] * df['Intensity']
        df['Grade'] = df['T'].apply(lambda x: 'A' if x>=16 else 'B' if x>=10 else 'C' if x>=5 else 'D' if x>=3 else 'E')
        return df
    except Exception as e:
        data = {
            'Activity': ['Steel Fixing', 'Concrete Pouring', 'Scaffolding'],
            'Hazard': ['Fall from height', 'Collapse', 'Structural failure'],
            'Improvement': ['Use safety harness\nDaily inspection', 'Proper shoring\nLimit load', 'Certified workers\nRegular check'],
            'Frequency': [3, 2, 4],
            'Intensity': [4, 5, 3],
            'T': [12, 10, 12],
            'Grade': ['B', 'B', 'B']
        }
        return pd.DataFrame(data)

def embed_texts(texts, api_key):
    openai.api_key = api_key
    return [openai.Embedding.create(input=t, model="text-embedding-3-small")['data'][0]['embedding'] for t in texts]

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index

# ------------------ GPT 처리 함수 ------------------
def generate_hazard(activity, examples, lang, api_key):
    prompt = f"Predict hazards for: {activity}\nExamples:\n" + "\n".join(examples)
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    ).choices[0].message.content

def generate_improvement(hazard, examples, lang, api_key):
    prompt = f"Generate improvements for: {hazard}\nExamples:\n" + "\n".join(examples)
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    ).choices[0].message.content

# ------------------ 메인 애플리케이션 ------------------
def main():
    st.set_page_config(layout="wide", page_title="AI Risk Assessment")
    
    # 언어 선택
    lang = st.sidebar.selectbox("Language", ["Korean", "English", "Chinese"])
    texts = system_texts[lang]
    
    # 데이터 초기화
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.index = None
    
    # 사이드바 설정
    with st.sidebar:
        st.header(texts['dataset_label'])
        dataset = st.selectbox("", ["Construction", "Civil", "Marine"])
        api_key = st.text_input(texts['api_key_label'], type="password")
        
        if st.button(texts['load_data_btn']):
            if not api_key:
                st.error(texts['error_api'])
            else:
                with st.spinner("Loading..."):
                    df = load_dataset(dataset)
                    embeddings = embed_texts(df['Activity'].tolist(), api_key)
                    st.session_state.data = df
                    st.session_state.index = build_faiss_index(embeddings)
    
    # 메인 콘텐츠
    st.markdown(f"<h1 class='main-title'>{texts['title']}</h1>", unsafe_allow_html=True)
    
    with st.container():
        activity = st.text_input(texts['activity_label'])
        
        if st.button(texts['predict_btn']):
            if not activity:
                st.error(texts['error_activity'])
            elif not st.session_state.data:
                st.error(texts['error_data'])
            else:
                with st.spinner("Analyzing..."):
                    # 유사 사례 검색
                    query_embed = embed_texts([activity], api_key)[0]
                    _, indices = st.session_state.index.search(np.array([query_embed], dtype='float32'), 3)
                    similar = st.session_state.data.iloc[indices[0]]
                    
                    # 위험 평가
                    hazard = generate_hazard(activity, similar['Hazard'].tolist(), lang, api_key)
                    
                    # 개선안 생성
                    improvement = generate_improvement(hazard, similar['Improvement'].tolist(), lang, api_key)
                    
                    # 결과 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {texts['similar_header']}")
                        for idx, row in similar.iterrows():
                            st.markdown(f"""
                            <div class='similar-case'>
                                <b>Case {idx+1}</b><br>
                                {texts['result_activity']} {row['Activity']}<br>
                                {texts['result_hazard']} {row['Hazard']}<br>
                                {texts['result_freq']} {row['Frequency']} | 
                                {texts['result_intensity']} {row['Intensity']}<br>
                                {texts['improve_header']}:<br>{row['Improvement']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"### {texts['risk_header']}")
                        st.markdown(f"""
                        <div class='result-box'>
                            <b>{texts['result_activity']}</b> {activity}<br>
                            <b>{texts['result_hazard']}</b> {hazard}<br><br>
                            <b>{texts['improve_header']}</b><br>
                            {improvement}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 위험 지표
                        st.markdown(f"### {texts['improve_header']}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(texts['improve_before'], "T: 12 (Grade B)")
                        with col_b:
                            st.metric(texts['improve_after'], "T: 4 (Grade D)", delta="-66.7%")

    # 푸터
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("cau_logo.png"):
            st.image("cau_logo.png", width=150)
    with col2:
        if os.path.exists("doosan_logo.png"):
            st.image("doosan_logo.png", width=180)

if __name__ == "__main__":
    main()
