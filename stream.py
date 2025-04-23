import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# 페이지 설정
st.set_page_config(
    page_title="AI 위험성평가 자동 생성 및 사고 예측",
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
</style>
""", unsafe_allow_html=True)

# 헤더 표시
st.markdown('<div class="main-header">AI 활용 위험성평가 자동 생성 및 사고 예측</div>', unsafe_allow_html=True)

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None

# 탭 설정
tabs = st.tabs(["시스템 개요", "위험성 평가 (Phase 1)", "개선대책 생성 (Phase 2)"])

# ------------------ 유틸리티 함수 ------------------

def determine_grade(value):
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


def load_data(selected_dataset_name):
    try:
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: '빈도', df.columns[5]: '강도'})
        df['T'] = pd.to_numeric(df.iloc[:,4]) * pd.to_numeric(df.iloc[:,5])
        df = df.iloc[:,:7]
        df.rename(
            columns={
                '작업활동 및 내용\nWork & Contents':'작업활동 및 내용',
                '유해위험요인 및 환경측면 영향\nHazard & Risk':'유해위험요인 및 환경측면 영향',
                '피해형태 및 환경영향\nDamage & Effect':'피해형태 및 환경영향'
            }, inplace=True)
        df = df.rename(columns={df.columns[6]:'T'})
        df['등급'] = df['T'].apply(determine_grade)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        st.warning("Excel 파일을 찾을 수 없어 샘플 데이터를 생성합니다.")
        data = {
            "작업활동 및 내용":["Shoring Installation","In and Out of materials","Transport / Delivery","Survey and Inspection"],
            "유해위험요인 및 환경측면 영향":["Fall and collision due to unstable ground","Overturning of transport vehicle","Collision between transport vehicle","Personnel fall while inspecting"],
            "피해형태 및 환경영향":["Injury","Equipment damage","Collision injury","Fall injury"],
            "빈도":[3,3,3,2],"강도":[2,3,5,3]
        }
        df = pd.DataFrame(data)
        df['T'] = df['빈도']*df['강도']
        df['등급'] = df['T'].apply(determine_grade)
        return df


def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    if api_key:
        openai.api_key = api_key
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)
    for idx, text in enumerate(texts):
        try:
            text = str(text).replace("\n"," ")
            response = openai.Embedding.create(model=model, input=[text])
            embeddings.append(response["data"][0]["embedding"])
        except:
            embeddings.append([0]*1536)
        progress_bar.progress((idx+1)/total)
    return embeddings


def generate_with_gpt(prompt, api_key=None, model="gpt-4o"):
    if api_key:
        openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"위험성 평가 및 개선대책 생성을 돕는 도우미입니다."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=250
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

# ----- Phase1 전용 Prompt/Parser (유해위험요인 예측) -----

def construct_prompt_phase1_for_hazard(retrieved_docs, query_activity):
    prompt = ""
    for i, row in enumerate(retrieved_docs.itertuples(),1):
        activity = getattr(row,'content')
        hazard = getattr(row,'유해위험요인 및 환경측면 영향')
        prompt += f"예시 {i}:\n입력: {activity}\n출력: {hazard}\n
"
    prompt += (
        f"입력: {query_activity}\n"
        "위 작업활동 및 내용을 바탕으로 유해위험요인을 한 문장으로 예측하세요.\n"
        "다음 JSON 형식으로 반환하세요:\n"
        '{"유해위험요인":"여기에 예측 결과"}\n'
    )
    return prompt


def parse_gpt_output_phase1_for_hazard(gpt_output):
    try:
        m = re.search(r'\{.*\}', gpt_output, re.DOTALL)
        if not m:
            return None
        data = re.json.loads(m.group())
        return data.get("유해위험요인")
    except:
        return None

# ----- Phase2: 개선대책 생성 관련 함수 -----
def compute_rrr(T_before, T_after):
    if T_before == 0:
        return 0.0
    return ((T_before - T_after) / T_before) * 100.0


def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    example_section = ""
    examples_added = 0
    for _, row in retrieved_docs.iterrows():
        try:
            improvement_plan = ""
            for field in ['개선대책 및 세부관리방안','개선대책','개선방안']:
                if field in row and pd.notna(row[field]):
                    improvement_plan = row[field]
                    break
            if not improvement_plan:
                continue
            orig_f = int(row['빈도'])
            orig_i = int(row['강도'])
            orig_T = orig_f * orig_i
            imp_f, imp_i, imp_T = 1,1,1
            for pat in [('개선 후 빈도','개선 후 강도','개선 후 T'),('개선빈도','개선강도','개선T')]:
                if all(p in row for p in pat):
                    imp_f = int(row[pat[0]]); imp_i = int(row[pat[1]]); imp_T = int(row[pat[2]]); break
            example_section += (
                "Example:\n"
                f"Input (Activity): {row['작업활동 및 내용']}\n"
                f"Input (Hazard): {row['유해위험요인 및 환경측면 영향']}\n"
                f"Input (Original Frequency): {orig_f}\n"
                f"Input (Original Intensity): {orig_i}\n"
                f"Input (Original T): {orig_T}\n"
                "Output (Improvement Plan and Risk Reduction) in JSON:\n"
                "{\n"
                f"  \"개선대책\": \"{improvement_plan}\",\n"
                f"  \"개선 후 빈도\": {imp_f},\n"
                f"  \"개선 후 강도\": {imp_i},\n"
                f"  \"개선 후 T\": {imp_T},\n"
                f"  \"T 감소율\": {compute_rrr(orig_T, imp_T):.2f}\n"
                "}\n\n"
            )
            examples_added += 1
            if examples_added >= 3:
                break
        except:
            continue
    if examples_added == 0:
        example_section = "... 기본 예시 ..."
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
        "Provide at least 3 specific improvement measures as a numbered list.\n"
        "Make sure to return only valid JSON.\n"
        "Output:\n"
    )
    return prompt


def parse_gpt_output_phase2(gpt_output):
    try:
        pattern = re.compile(r"```json(.*?)```", re.DOTALL)
        m = pattern.search(gpt_output)
        json_str = m.group(1).strip() if m else gpt_output.replace("```","")
        return pd.json.loads(json_str)
    except Exception as e:
        st.error(f"JSON 파싱 중 오류 발생: {str(e)}")
        return None

# 데이터셋 옵션
dataset_options = {
    "SWRO 건축공정 (건축)":"SWRO 건축공정 (건축)",
    "Civil (토목)":"Civil (토목)",
    "Marine (토목)":"Marine (토목)",
    "SWRO 기계공사 (플랜트)":"SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)":"SWRO 전기작업표준 (플랜트)"
}

# 시스템 개요 탭
with tabs[0]:
    st.markdown('<div class="sub-header">LLM 기반 위험성평가 시스템</div>', unsafe_allow_html=True)
    # ... 개요 내용 생략 가능

# Phase 1 탭
with tabs[1]:
    st.markdown('<div class="sub-header">위험성 평가 자동화 (Phase 1)</div>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password", key="api_key_phase1")
    selected_dataset_name = st.selectbox("데이터셋 선택", list(dataset_options.keys()), key="dataset_selector_phase1")
    if st.button("데이터 로드 및 인덱스 구성", key="load_data_phase1"):
        if not api_key:
            st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        else:
            with st.spinner('데이터를 불러오고 인덱스를 구성하는 중...'):
                df = load_data(dataset_options[selected_dataset_name])
                train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
                retriever_pool_df = train_df.copy()
                retriever_pool_df['content'] = retriever_pool_df['작업활동 및 내용'].astype(str)
                texts = retriever_pool_df['content'].tolist()
                max_texts = min(len(texts), 10)
                st.info(f"데모: {max_texts}개 텍스트 임베딩 처리")
                embeddings = embed_texts_with_openai(texts[:max_texts], api_key=api_key)
                embeddings_array = np.array(embeddings, dtype='float32')
                faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
                faiss_index.add(embeddings_array)
                st.session_state.index = faiss_index
                st.session_state.embeddings = embeddings_array
                st.session_state.retriever_pool_df = retriever_pool_df.iloc[:max_texts]
                st.success("데이터 로드 및 인덱스 구성 완료!")
    if st.session_state.index is None:
        st.warning("먼저 인덱스를 구성하세요.")
    else:
        with st.form("user_input_form"):
            user_activity = st.text_input("작업활동 및 내용:", key="form_user_activity")
            submitted = st.form_submit_button("유해위험요인 예측하기")
        if submitted:
            if not user_activity:
                st.warning("작업활동 및 내용을 입력하세요.")
            else:
                with st.spinner("예측 중..."):
                    query_embedding = embed_texts_with_openai([user_activity], api_key=api_key)[0]
                    distances, indices = st.session_state.index.search(np.array([query_embedding],dtype='float32'), 3)
                    retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]
                    prompt = construct_prompt_phase1_for_hazard(retrieved_docs, user_activity)
                    generated_output = generate_with_gpt(prompt, api_key=api_key)
                    hazard_pred = parse_gpt_output_phase1_for_hazard(generated_output)
                    st.markdown(f"**작업활동 및 내용:** {user_activity}")
                    st.markdown(f"**예측된 유해위험요인:** {hazard_pred}")
                    st.markdown("#### 유사 사례")
                    for _, row in retrieved_docs.iterrows():
                        st.markdown(f"- **작업활동 및 내용:** {row['작업활동 및 내용']}")
                        st.markdown(f"  - 유해위험요인: {row['유해위험요인 및 환경측면 영향']}")

# Phase2 탭 전체 코드
with tabs[2]:
    st.markdown('<div class="sub-header">개선대책 자동 생성 (Phase 2)</div>', unsafe_allow_html=True)
    api_key_phase2 = st.text_input("OpenAI API 키를 입력하세요:", type="password", key="api_key_phase2")
    target_language = st.selectbox(
        "개선대책 언어 선택:",
        options=["Korean", "English", "Chinese"],
        index=0,
        key="target_language"
    )
    input_method = st.radio(
        "입력 방식 선택:",
        options=["Phase 1 평가 결과 사용", "직접 입력"],
        index=0,
        key="input_method"
    )
    if input_method == "Phase 1 평가 결과 사용":
        if hasattr(st.session_state, 'last_assessment'):
            last_assessment = st.session_state.last_assessment
            st.markdown("### Phase 1 평가 결과")
            st.markdown(f"**작업활동:** {last_assessment['activity']}")
            st.markdown(f"**유해위험요인:** {last_assessment['hazard']}")
            st.markdown(
                f"**위험도:** 빈도 {last_assessment['frequency']}, 강도 {last_assessment['intensity']}, T값 {last_assessment['T']} (등급 {last_assessment['grade']})"
            )
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
        col1, col2 = st.columns(2)
        with col1:
            activity_text = st.text_input("작업활동:", key="direct_activity")
            hazard_text = st.text_input("유해위험요인:", key="direct_hazard")
        with col2:
            frequency = st.number_input("빈도 (1-5):", min_value=1, max_value=5, value=3, key="direct_freq")
            intensity = st.number_input("강도 (1-5):", min_value=1, max_value=5, value=3, key="direct_intensity")
            T_value = frequency * intensity
            st.markdown(f"**T값:** {T_value} (등급: {determine_grade(T_value)})")
    if st.button("개선대책 생성", key="generate_improvement") and activity_text and hazard_text:
        if not api_key_phase2:
            st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        else:
            with st.spinner("개선대책을 생성하는 중..."):
                if st.session_state.retriever_pool_df is None or st.session_state.index is None:
                    df = load_data("Civil (토목)")
                    retriever_pool_df = df.sample(min(5, len(df)))
                    retrieved_docs = retriever_pool_df.sample(min(3, len(retriever_pool_df)))
                else:
                    retriever_pool_df = st.session_state.retriever_pool_df
                    query_text = f"{activity_text} {hazard_text}"
                    query_embedding = embed_texts_with_openai([query_text], api_key=api_key_phase2)[0]
                    distances, indices = st.session_state.index.search(np.array([query_embedding], dtype='float32'), 3)
                    retrieved_docs = retriever_pool_df.iloc[indices[0]]
                prompt = construct_prompt_phase2(
                    retrieved_docs, activity_text, hazard_text,
                    frequency, intensity, T_value, target_language
                )
                generated_output = generate_with_gpt(prompt, api_key=api_key_phase2)
                parsed = parse_gpt_output_phase2(generated_output)
                if parsed:
                    improvement_plan = parsed.get("개선대책", "")
                    imp_freq = parsed.get("개선 후 빈도", 1)
                    imp_int = parsed.get("개선 후 강도", 1)
                    imp_T = parsed.get("개선 후 T", imp_freq*imp_int)
                    rrr = parsed.get("T 감소율", compute_rrr(T_value, imp_T))
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("#### 개선대책 생성 결과")
                    c1, c2 = st.columns([3,2])
                    with c1:
                        st.markdown("##### 개선대책")
                        st.markdown(improvement_plan)
                    with c2:
                        comp_df = pd.DataFrame({
                            '항목':['빈도','강도','T값','위험등급'],
                            '개선 전':[frequency,intensity,T_value,determine_grade(T_value)],
                            '개선 후':[imp_freq,imp_int,imp_T,determine_grade(imp_T)]
                        })
                        st.table(comp_df)
                        st.metric("위험 감소율 (RRR)", f"{rrr:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("#### 위험도(T값) 변화")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**개선 전 T값:**")
                        st.progress(T_value/25)
                    with col2:
                        st.markdown("**개선 후 T값:**")
                        st.progress(imp_T/25)
                else:
                    st.error("개선대책 생성 결과를 파싱할 수 없습니다.")
                    st.write(generated_output)

# 푸터
st.markdown('<hr style="margin-top:50px;">', unsafe_allow_html=True)
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("cau.png"): st.image(Image.open("cau.png"), width=150)
with col2:
    if os.path.exists("doosan.png"): st.image(Image.open("doosan.png"), width=180)
st.markdown('</div>', unsafe_allow_html=True)
