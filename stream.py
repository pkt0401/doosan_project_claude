import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from sklearn.model_selection import train_test_split

# 페이지 설정
st.set_page_config(
    page_title="LLM 활용 위험성평가 자동 생성 및 사고 예측",
    page_icon="🛠️",
    layout="wide"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
    }
    .phase-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    .process-step {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .process-arrow {
        color: #4CAF50;
        font-size: 20px;
        margin: 0 10px;
    }
    .process-container {
        display: flex;
        flex-direction: column;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin: 20px 0;
    }
    .step-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .step-number {
        margin-right: 10px;
        font-weight: bold;
    }
    .step-content {
        flex-grow: 1;
    }
    .arrow-down {
        text-align: center;
        margin: 5px 0;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# 헤더 표시
st.markdown('<div class="main-header">LLM 활용 위험성평가 자동 생성 및 사고 예측</div>', unsafe_allow_html=True)

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None

# 탭 설정 - 데이터 탐색 탭 제거
tabs = st.tabs(["시스템 개요", "위험성 평가 (Phase 1)", "개선대책 생성 (Phase 2)"])

# ------------------ 유틸리티 함수 ------------------

# 빈도*강도 결과 T에 따른 등급 결정 함수
def determine_grade(value):
    """빈도*강도 결과 T에 따른 등급 결정 함수."""
    if 16 <= value <= 25:
        return 'A'
    elif 10 <= value <= 15:
        return 'B'
    elif 5 <= value <= 9:
        return 'C'
    elif 3 <= value <= 4:
        return 'D'
    elif 1 <= value <= 2:
        return 'E'
    else:
        return '알 수 없음'

# 데이터 불러오기 함수
def load_data(selected_dataset_name):
    """선택된 이름에 대응하는 Excel 데이터 불러오기."""
    try:
        # 실제 Excel 파일에서 데이터 로드
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")

        # 전처리
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: '빈도'})
        df = df.rename(columns={df.columns[5]: '강도'})

        df['T'] = pd.to_numeric(df.iloc[:, 4]) * pd.to_numeric(df.iloc[:, 5])
        df = df.iloc[:, :7]
        df.rename(
            columns={
                '작업활동 및 내용\nWork & Contents': '작업활동 및 내용',
                '유해위험요인 및 환경측면 영향\nHazard & Risk': '유해위험요인 및 환경측면 영향',
                '피해형태 및 환경영향\nDamage & Effect': '피해형태 및 환경영향'
            },
            inplace=True
        )
        df = df.rename(columns={df.columns[6]: 'T'})
        df['등급'] = df['T'].apply(determine_grade)

        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        st.write(f"시도한 파일 경로: {selected_dataset_name}")
        
        # 파일이 없는 경우 대비한 더미 데이터 생성 (데모용)
        st.warning("Excel 파일을 찾을 수 없어 샘플 데이터를 생성합니다. 실제 운영 시에는 해당 파일이 필요합니다.")
        data = {
            "작업활동 및 내용": ["Shoring Installation", "In and Out of materials", "Transport / Delivery", "Survey and Inspection"],
            "유해위험요인 및 환경측면 영향": ["Fall and collision due to unstable ground", "Overturning of transport vehicle", 
                                 "Collision between transport vehicle", "Personnel fall while inspecting"],
            "피해형태 및 환경영향": ["Injury", "Equipment damage", "Collision injury", "Fall injury"],
            "빈도": [3, 3, 3, 2],
            "강도": [2, 3, 5, 3]
        }
        
        df = pd.DataFrame(data)
        df['T'] = df['빈도'] * df['강도']
        df['등급'] = df['T'].apply(determine_grade)
        
        return df

# OpenAI 임베딩 API를 통해 텍스트 임베딩 생성
def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    """OpenAI 임베딩 API로 텍스트 리스트를 임베딩."""
    if api_key:
        openai.api_key = api_key
    
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)

    for idx, text in enumerate(texts):
        try:
            text = str(text).replace("\n", " ")
            response = openai.Embedding.create(model=model, input=[text])
            embedding = response["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"텍스트 임베딩 중 오류 발생: {str(e)}")
            embeddings.append([0]*1536)
        
        progress_bar.progress((idx + 1) / total)
    
    return embeddings

# GPT 모델을 통해 예측 결과 생성
def generate_with_gpt(prompt, api_key=None, model="gpt-4o"):
    """GPT 모델로부터 예측 결과를 받아오는 함수."""
    if api_key:
        openai.api_key = api_key
        
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "위험성 평가 및 개선대책 생성을 돕는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=250
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

# ----- Phase 1: 위험성 평가 관련 함수 -----

# 검색된 문서로 GPT 프롬프트 생성 (Phase 1)
def construct_prompt_phase1(retrieved_docs, query_text):
    """검색된 문서들로부터 예시를 구성해 GPT 프롬프트 생성."""
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            example_input = f"{doc['작업활동 및 내용']} {doc['유해위험요인 및 환경측면 영향']}"
            frequency = int(doc['빈도'])
            intensity = int(doc['강도'])
            T_value = frequency * intensity
            example_output = f'{{"빈도": {frequency}, "강도": {intensity}, "T": {T_value}}}'
            retrieved_examples.append((example_input, example_output))
        except:
            continue
    
    prompt = ""
    for i, (example_input, example_output) in enumerate(retrieved_examples, 1):
        prompt += f"예시 {i}:\n입력: {example_input}\n출력: {example_output}\n\n"
    
    prompt += (
        f"입력: {query_text}\n"
        "위 입력을 바탕으로 빈도와 강도를 예측하세요. "
        "빈도는 1에서 5 사이의 정수입니다. "
        "강도는 1에서 5 사이의 정수입니다. "
        "T는 빈도와 강도를 곱한 값입니다.\n"
        "다음 JSON 형식으로 출력하세요:\n"
        '{"빈도": 숫자, "강도": 숫자, "T": 숫자}\n'
        "출력:\n"
    )
    return prompt

# GPT 출력 파싱 (Phase 1)
def parse_gpt_output_phase1(gpt_output):
    """GPT 출력에서 {빈도, 강도, T}를 정규표현식으로 추출."""
    json_pattern = r'\{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    match = re.search(json_pattern, gpt_output)
    if match:
        pred_frequency = int(match.group(1))
        pred_intensity = int(match.group(2))
        pred_T = int(match.group(3))
        return pred_frequency, pred_intensity, pred_T
    else:
        return None

# ----- Phase 2: 개선대책 생성 관련 함수 -----

# 위험 감소율(RRR) 계산 함수
def compute_rrr(T_before, T_after):
    """위험 감소율(Risk Reduction Rate) 계산"""
    if T_before == 0:
        return 0.0
    return ((T_before - T_after) / T_before) * 100.0

# 개선대책 생성을 위한 프롬프트 구성 (Phase 2)
def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    """
    개선대책 생성을 위한 프롬프트 구성
    """
    # 예시 섹션 구성
    example_section = ""
    examples_added = 0
    
    for _, row in retrieved_docs.iterrows():
        try:
            # Phase2 데이터셋에 있는 개선대책 필드 사용 시도
            # 여러 가능한 필드명 시도
            improvement_plan = ""
            for field in ['개선대책 및 세부관리방안', '개선대책', '개선방안']:
                if field in row and pd.notna(row[field]):
                    improvement_plan = row[field]
                    break
            
            if not improvement_plan:
                continue  # 개선대책이 없으면 건너뛰기
                
            original_freq = int(row['빈도']) if '빈도' in row else 3
            original_intensity = int(row['강도']) if '강도' in row else 3
            original_T = original_freq * original_intensity
                
            # 개선 후 데이터 시도
            improved_freq = 1
            improved_intensity = 1
            improved_T = 1
            
            for field_pattern in [('개선 후 빈도', '개선 후 강도', '개선 후 T'), ('개선빈도', '개선강도', '개선T')]:
                if all(field in row for field in field_pattern):
                    improved_freq = int(row[field_pattern[0]])
                    improved_intensity = int(row[field_pattern[1]])
                    improved_T = int(row[field_pattern[2]])
                    break
            
            example_section += (
                "Example:\n"
                f"Input (Activity): {row['작업활동 및 내용']}\n"
                f"Input (Hazard): {row['유해위험요인 및 환경측면 영향']}\n"
                f"Input (Original Frequency): {original_freq}\n"
                f"Input (Original Intensity): {original_intensity}\n"
                f"Input (Original T): {original_T}\n"
                "Output (Improvement Plan and Risk Reduction) in JSON:\n"
                "{\n"
                f"  \"개선대책\": \"{improvement_plan}\",\n"
                f"  \"개선 후 빈도\": {improved_freq},\n"
                f"  \"개선 후 강도\": {improved_intensity},\n"
                f"  \"개선 후 T\": {improved_T},\n"
                f"  \"T 감소율\": {compute_rrr(original_T, improved_T):.2f}\n"
                "}\n\n"
            )
            
            examples_added += 1
            if examples_added >= 3:  # 최대 3개 예시만 사용
                break
                
        except Exception as e:
            # 에러 발생 시 해당 예시 건너뛰기
            continue
    
    # 예시가 없는 경우 기본 예시 추가
    if examples_added == 0:
        example_section = """
Example:
Input (Activity): Excavation and backfilling
Input (Hazard): Collapse of excavation wall due to improper sloping
Input (Original Frequency): 3
Input (Original Intensity): 4
Input (Original T): 12
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "개선대책": "1) 토양 분류에 따른 적절한 경사 유지 2) 굴착 벽면 보강 3) 정기적인 지반 상태 검사 실시",
  "개선 후 빈도": 1,
  "개선 후 강도": 2,
  "개선 후 T": 2,
  "T 감소율": 83.33
}

Example:
Input (Activity): Lifting operation
Input (Hazard): Material fall due to improper rigging
Input (Original Frequency): 2
Input (Original Intensity): 5
Input (Original T): 10
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "개선대책": "1) 리깅 전문가 작업 참여 2) 리깅 장비 사전 점검 3) 안전 구역 설정 및 접근 통제",
  "개선 후 빈도": 1,
  "개선 후 강도": 2,
  "개선 후 T": 2,
  "T 감소율": 80.00
}
"""
    
    # 최종 프롬프트 생성
    prompt = (
        f"{example_section}"
        "Now here is a new input:\n"
        f"Input (Activity): {activity_text}\n"
        f"Input (Hazard): {hazard_text}\n"
        f"Input (Original Frequency): {freq}\n"
        f"Input (Original Intensity): {intensity}\n"
        f"Input (Original T): {T}\n\n"
        "Please provide the output in JSON format with these keys:\n"
        "{\n"
        "  \"개선대책\": \"항목별 개선대책 리스트\", \n"
        "  \"개선 후 빈도\": (an integer in [1..5]),\n"
        "  \"개선 후 강도\": (an integer in [1..5]),\n"
        "  \"개선 후 T\": (Improved Frequency * Improved Severity),\n"
        "  \"T 감소율\": (percentage of risk reduction)\n"
        "}\n\n"
        f"Please write the improvement measures (개선대책) in {target_language}.\n"
        f"Provide at least 3 specific improvement measures as a numbered list.\n"
        "Make sure to return only valid JSON.\n"
        "Output:\n"
    )
    
    return prompt

# GPT 응답 파싱 (Phase 2)
def parse_gpt_output_phase2(gpt_output):
    """GPT 응답에서 JSON 데이터를 추출"""
    try:
        # JSON 블록이 있는 경우 추출 시도
        pattern = re.compile(r"```json(.*?)```", re.DOTALL)
        match = pattern.search(gpt_output)

        if match:
            json_str = match.group(1).strip()
        else:
            # JSON 블록 표시가 없는 경우 원문을 JSON으로 파싱 시도
            json_str = gpt_output.replace("```", "").strip()

        import json
        result = json.loads(json_str)
        return result
    except Exception as e:
        st.error(f"JSON 파싱 중 오류 발생: {str(e)}")
        st.write("원본 GPT 응답:", gpt_output)
        return None

# ------------------ 데이터셋 및 샘플 데이터 준비 ------------------

# 데이터셋 옵션
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "Marine (토목)": "Marine (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)": "SWRO 전기작업표준 (플랜트)"
}

# ----- 시스템 개요 탭 -----
with tabs[0]:
    st.markdown('<div class="sub-header">LLM 기반 위험성평가 시스템</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-text">
        LLM(Large Language Model)을 활용한 위험성평가 자동화 시스템은 건설 현장의 안전 관리를 혁신적으로 개선합니다:
        
        1. <span class="highlight">작업 내용 입력 시 생성형 AI를 통한 '유해위험요인' 및 '위험 등급' 자동 생성</span> <span class="phase-badge">Phase 1</span>
        2. <span class="highlight">위험도 감소를 위한 개선대책 자동 생성 및 감소율 예측</span> <span class="phase-badge">Phase 2</span>
        3. AI는 건설현장의 기존 위험성평가를 공정별로 구분하고, 해당 유해위험요인을 학습
        4. 자동 생성 기술 개발 완료 후 위험도 기반 사고위험성과 이미지 분석을 통한 사고예측 (계획)
        
        이 시스템은 PIMS 및 안전지킴이 등 EHS 플랫폼에 AI 기술 탑재를 통해 통합 사고 예측 프로그램으로 발전 예정입니다.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # AI 위험성평가 프로세스 다이어그램 - 수정된 버전
        st.markdown('<div style="text-align: center; margin-bottom: 10px;"><b>AI 위험성평가 프로세스</b></div>', unsafe_allow_html=True)
        
        # 프로세스 단계 및 해당 Phase 지정
        steps = ["작업내용 입력", "AI 위험분석", "유해요인 식별", "위험등급 산정", "개선대책 자동생성", "안전조치 적용"]
        phases = ["Phase 1", "Phase 1", "Phase 1", "Phase 1", "Phase 2", "Phase 2"]
        
        # 프로세스 흐름을 수직 방향으로 표현
        st.markdown('<div class="process-container">', unsafe_allow_html=True)
        
        for i, (step, phase) in enumerate(zip(steps, phases)):
            st.markdown(
                f'<div class="step-container">'
                f'<div class="step-number">{i+1}.</div>'
                f'<div class="step-content">{step} <span class="phase-badge">{phase}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # 마지막 단계가 아니면 화살표 추가
            if i < len(steps) - 1:
                st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 시스템 특징
    st.markdown('<div class="sub-header">시스템 특징 및 구성요소</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Phase 1: 위험성 평가 자동화
        - 공정별 작업활동에 따른 위험성평가 데이터 학습
        - 작업활동 입력 시 유해위험요인 및 위험도 자동 예측
        - 대규모 언어 모델(LLM) 기반 위험도(빈도, 강도, T) 측정
        - Excel 기반 공정별 위험성평가 데이터 자동 분석
        - 유사 사례 검색 및 위험등급(A-E) 자동 산정
        """)
    
    with col2:
        st.markdown("""
        #### Phase 2: 개선대책 자동 생성
        - 위험요소별 맞춤형 개선대책 자동 생성
        - 다국어(한/영/중) 개선대책 생성 지원
        - 개선 전후 위험도(T) 자동 비교 분석
        - 위험 감소율(RRR) 정량적 산출
        - 공종/공정별 최적 개선대책 데이터베이스 구축
        - 두산 건설현장 통합 AI 안전관리 시스템 연계
        """)

# ----- Phase 1: 위험성 평가 탭 -----
with tabs[1]:
    st.markdown('<div class="sub-header">위험성 평가 자동화 (Phase 1)</div>', unsafe_allow_html=True)
    
    # API 키 입력
    api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password", key="api_key_phase1")
    
    # 데이터셋 선택
    selected_dataset_name = st.selectbox(
        "데이터셋 선택",
        options=list(dataset_options.keys()),
        key="dataset_selector_phase1"
    )
    
    # 인덱스 구성 섹션
    st.markdown("### 1. 데이터 로드 및 인덱스 구성")
    
    if st.button("데이터 로드 및 인덱스 구성", key="load_data_phase1"):
        if not api_key:
            st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        else:
            with st.spinner('데이터를 불러오고 인덱스를 구성하는 중...'):
                # 데이터 불러오기
                df = load_data(dataset_options[selected_dataset_name])
                
                if df is not None:
                    # Train/Test 분할
                    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
                    
                    # 리트리버 풀 구성
                    retriever_pool_df = train_df.copy()
                    retriever_pool_df['content'] = retriever_pool_df.apply(
                        lambda row: ' '.join(row.values.astype(str)), axis=1
                    )
                    texts = retriever_pool_df['content'].tolist()
                    
                    # 임베딩 생성 (데모에서는 적은 수만 처리, 실제로는 전체 처리)
                    max_texts = min(len(texts), 10)  # 데모에서는 최대 10개만 처리
                    st.info(f"데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.")
                    
                    openai.api_key = api_key
                    embeddings = embed_texts_with_openai(texts[:max_texts], api_key=api_key)
                    
                    # FAISS 인덱스 구성
                    embeddings_array = np.array(embeddings, dtype='float32')
                    dimension = embeddings_array.shape[1]
                    faiss_index = faiss.IndexFlatL2(dimension)
                    faiss_index.add(embeddings_array)
                    
                    # 세션 상태에 저장
                    st.session_state.index = faiss_index
                    st.session_state.embeddings = embeddings_array
                    st.session_state.retriever_pool_df = retriever_pool_df.iloc[:max_texts]  # 임베딩된 부분만 저장
                    
                    st.success(f"데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)")
                    st.session_state.test_df = test_df
    
    # 사용자 입력 예측 섹션
    st.markdown("### 2. 위험성 평가 예측")
    
    if st.session_state.index is None:
        st.warning("먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.")
    else:
        with st.form("user_input_form"):
            user_work = st.text_input("작업활동:", key="form_user_work")
            user_risk = st.text_input("유해위험요인:", key="form_user_risk")
            submitted = st.form_submit_button("위험성 평가 예측하기")
            
        if submitted:
            if not user_work or not user_risk:
                st.warning("작업활동과 유해위험요인을 모두 입력하세요.")
            else:
                query_text = f"{user_work} {user_risk}"
                
                with st.spinner("위험성을 평가하는 중..."):
                    # 쿼리 임베딩
                    query_embedding = embed_texts_with_openai([query_text], api_key=api_key)[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    
                    # 유사 문서 검색
                    k_similar = min(3, len(st.session_state.retriever_pool_df))
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]
                    
                    # GPT 프롬프트 생성 & 호출
                    prompt = construct_prompt_phase1(retrieved_docs, query_text)
                    generated_output = generate_with_gpt(prompt, api_key=api_key)
                    
                    # 결과 표시
                    st.markdown(f"**사용자 입력 작업활동:** {user_work}")
                    st.markdown(f"**사용자 입력 유해위험요인:** {user_risk}")
                    
                    parse_result = parse_gpt_output_phase1(generated_output)
                    if parse_result is not None:
                        f_val, i_val, t_val = parse_result
                        grade = determine_grade(t_val)
                        
                        # 결과를 표로 표시
                        result_df = pd.DataFrame({
                            '항목': ['빈도', '강도', 'T 값', '위험등급'],
                            '값': [f_val, i_val, t_val, grade]
                        })
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("#### 위험성 평가 결과")
                        st.table(result_df)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 세션 상태에 결과 저장 (Phase 2에서 사용)
                        st.session_state.last_assessment = {
                            'activity': user_work,
                            'hazard': user_risk,
                            'frequency': f_val,
                            'intensity': i_val,
                            'T': t_val,
                            'grade': grade
                        }
                    else:
                        st.error("위험성 평가 결과를 파싱할 수 없습니다.")
                        st.write(f"GPT 원문 응답: {generated_output}")

# ----- Phase 2: 개선대책 생성 탭 -----
with tabs[2]:
    st.markdown('<div class="sub-header">개선대책 자동 생성 (Phase 2)</div>', unsafe_allow_html=True)
    
    # API 키 입력
    api_key_phase2 = st.text_input("OpenAI API 키를 입력하세요:", type="password", key="api_key_phase2")
    
    # 개선대책 언어 선택
    target_language = st.selectbox(
        "개선대책 언어 선택:",
        options=["Korean", "English", "Chinese"],
        index=0,
        key="target_language"
    )
    
    # 입력 방식 선택 (Phase 1 결과 사용 또는 직접 입력)
    input_method = st.radio(
        "입력 방식 선택:",
        options=["Phase 1 평가 결과 사용", "직접 입력"],
        index=0,
        key="input_method"
    )
    
    if input_method == "Phase 1 평가 결과 사용":
        # Phase 1 결과를 사용하는 경우
        if hasattr(st.session_state, 'last_assessment'):
            last_assessment = st.session_state.last_assessment
            
            st.markdown("### Phase 1 평가 결과")
            st.markdown(f"**작업활동:** {last_assessment['activity']}")
            st.markdown(f"**유해위험요인:** {last_assessment['hazard']}")
            st.markdown(f"**위험도:** 빈도 {last_assessment['frequency']}, 강도 {last_assessment['intensity']}, T값 {last_assessment['T']} (등급 {last_assessment['grade']})")
            
            activity_text = last_assessment['activity']
            hazard_text = last_assessment['hazard']
            frequency = last_assessment['frequency']
            intensity = last_assessment['intensity']
            T_value = last_assessment['T']
            
        else:
            st.warning("먼저 Phase 1에서 위험성 평가를 수행하세요.")
            activity_text = hazard_text = None
            frequency = intensity = T_value = None
    else:
        # 직접 입력하는 경우
        col1, col2 = st.columns(2)
        
        with col1:
            activity_text = st.text_input("작업활동:", key="direct_activity")
            hazard_text = st.text_input("유해위험요인:", key="direct_hazard")
        
        with col2:
            frequency = st.number_input("빈도 (1-5):", min_value=1, max_value=5, value=3, key="direct_freq")
            intensity = st.number_input("강도 (1-5):", min_value=1, max_value=5, value=3, key="direct_intensity")
            T_value = frequency * intensity
            st.markdown(f"**T값:** {T_value} (등급: {determine_grade(T_value)})")
    
    # 개선대책 생성 섹션
    if st.button("개선대책 생성", key="generate_improvement") and activity_text and hazard_text and frequency and intensity and T_value:
        if not api_key_phase2:
            st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        else:
            with st.spinner("개선대책을 생성하는 중..."):
                # 리트리버 풀과 인덱스 확인
                if st.session_state.retriever_pool_df is None or st.session_state.index is None:
                    st.warning("Phase 1에서 데이터 로드 및 인덱스 구성을 완료하지 않았습니다. 기본 예시를 사용합니다.")
                    # 기본 공통 데이터셋 로드
                    df = load_data("Civil (토목)")
                    retriever_pool_df = df.sample(min(5, len(df)))  # 최대 5개 샘플
                    
                    # 유사 문서는 랜덤 샘플링
                    retrieved_docs = retriever_pool_df.sample(min(3, len(retriever_pool_df)))
                else:
                    # Phase 1에서 구성된 리트리버 풀 및 인덱스 사용
                    retriever_pool_df = st.session_state.retriever_pool_df
                    
                    # 유사 문서 검색 (실제 또는 모의)
                    k_similar = min(3, len(retriever_pool_df))
                    query_text = f"{activity_text} {hazard_text}"
                    query_embedding = embed_texts_with_openai([query_text], api_key=api_key_phase2)[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = retriever_pool_df.iloc[indices[0]]
                
                # 개선대책 생성 프롬프트 구성
                prompt = construct_prompt_phase2(
                    retrieved_docs, 
                    activity_text, 
                    hazard_text, 
                    frequency, 
                    intensity, 
                    T_value, 
                    target_language
                )
                
                # GPT 호출
                generated_output = generate_with_gpt(prompt, api_key=api_key_phase2)
                
                # 결과 파싱
                parsed_result = parse_gpt_output_phase2(generated_output)
                
                if parsed_result:
                    # 결과 표시
                    improvement_plan = parsed_result.get("개선대책", "")
                    improved_freq = parsed_result.get("개선 후 빈도", 1)
                    improved_intensity = parsed_result.get("개선 후 강도", 1)
                    improved_T = parsed_result.get("개선 후 T", improved_freq * improved_intensity)
                    rrr = parsed_result.get("T 감소율", compute_rrr(T_value, improved_T))
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("#### 개선대책 생성 결과")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("##### 개선대책")
                        st.markdown(improvement_plan)
                    
                    with col2:
                        st.markdown("##### 위험도 개선 결과")
                        
                        # 개선 전후 위험도 비교표
                        comparison_df = pd.DataFrame({
                            '항목': ['빈도', '강도', 'T값', '위험등급'],
                            '개선 전': [frequency, intensity, T_value, determine_grade(T_value)],
                            '개선 후': [improved_freq, improved_intensity, improved_T, determine_grade(improved_T)]
                        })
                        st.table(comparison_df)
                        
                        # 위험 감소율 표시
                        st.metric(
                            label="위험 감소율 (RRR)",
                            value=f"{rrr:.2f}%"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 위험도 그래프로 표현 (간단한 텍스트 기반 시각화)
                    st.markdown("#### 위험도(T값) 변화")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**개선 전 T값:**")
                        st.progress(T_value / 25)  # 25는 최대 T값
                    
                    with col2:
                        st.markdown("**개선 후 T값:**")
                        st.progress(improved_T / 25)
                else:
                    st.error("개선대책 생성 결과를 파싱할 수 없습니다.")
                    st.write("GPT 원문 응답:", generated_output)

# 푸터
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023-2025 두산건설 LLM 활용 위험성평가 자동 생성 시스템</p>
    <p style="font-size: 0.8rem;">최신 업데이트: 2025년 4월 8일</p>
</div>
""", unsafe_allow_html=True)
