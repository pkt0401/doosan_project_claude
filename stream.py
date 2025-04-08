import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import os
import json
from tqdm import tqdm

# ----- 전역 변수 (데이터셋 선택 옵션) -----
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "Marine (토목)": "Marine (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)": "SWRO 전기작업표준 (플랜트)"
}

# 언어 선택 옵션 (Phase 2에서 사용)
language_options = ["Korean", "Chinese", "English"]

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

def load_data(selected_dataset_name):
    """선택된 이름에 대응하는 Excel 데이터 불러오기."""
    try:
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
        return None

def generate_with_gpt4(prompt):
    """GPT-4 모델로부터 예측 결과를 받아오는 함수."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "위험성 평가 값을 예측하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT-4 API 호출 중 오류 발생: {str(e)}")
        return None

def embed_texts_with_openai(texts, model="text-embedding-3-large"):
    """OpenAI 임베딩 API로 텍스트 리스트를 임베딩."""
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)

    for idx, text in enumerate(tqdm(texts, desc="임베딩 진행 중", unit="개")):
        try:
            text = text.replace("\n", " ")
            response = openai.Embedding.create(model=model, input=[text])
            embedding = response["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"텍스트 임베딩 중 오류 발생: {str(e)}")
            embeddings.append([0]*1536)
        
        progress_bar.progress((idx + 1) / total)
    
    return embeddings

def construct_prompt(retrieved_docs, query_text):
    """검색된 문서들로부터 예시를 구성해 GPT 프롬프트 생성."""
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        content_parts = doc['content'].split()
        example_input = ' '.join(content_parts[:-6])
        frequency = int(content_parts[-4])
        intensity = int(content_parts[-3])
        T_value = frequency * intensity
        example_output = f'{{"빈도": {frequency}, "강도": {intensity}, "T": {T_value}}}'
        retrieved_examples.append((example_input, example_output))
    
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

def parse_gpt_output(gpt_output):
    """
    GPT 출력에서 {빈도, 강도, T}를 정규표현식으로 추출.
    매칭 성공 시 (빈도, 강도, T)를 리턴, 실패 시 None 리턴.
    """
    json_pattern = r'\{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    match = re.search(json_pattern, gpt_output)
    if match:
        pred_frequency = int(match.group(1))
        pred_intensity = int(match.group(2))
        pred_T = int(match.group(3))
        return pred_frequency, pred_intensity, pred_T
    else:
        return None

# ----- Phase 2 관련 추가 함수들 -----

def get_openai_embedding(text, model_name="text-embedding-3-small"):
    """OpenAI API를 사용해 단일 텍스트의 임베딩 벡터를 반환합니다."""
    response = openai.Embedding.create(
        model=model_name,
        input=[text]
    )
    vector = response["data"][0]["embedding"]
    return np.array(vector, dtype=np.float32)

def call_gpt(prompt, model_name="gpt-4o-mini"):
    """GPT 모델에 프롬프트를 전달하고 응답을 받아옵니다."""
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in safety enhancements to effectively minimize risk (T)."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {e}")
        return ""

def parse_gpt_response(raw_text):
    """GPT 응답에서 JSON 형식의 데이터를 추출합니다."""
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    match = pattern.search(raw_text)

    if match:
        json_str = match.group(1).strip()
    else:
        json_str = raw_text.replace("```", "").replace("```json", "").strip()

    try:
        return json.loads(json_str)
    except:
        try:
            # JSON 문법 오류가 있을 수 있어 더 관대한 방식으로 다시 시도
            import ast
            # 따옴표 통일 (작은따옴표를 큰따옴표로)
            fixed_str = json_str.replace("'", "\"")
            # 따옴표가 누락된 키를 수정
            fixed_str = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', fixed_str)
            return json.loads(fixed_str)
        except:
            st.warning("GPT 응답을 JSON으로 파싱하는데 실패했습니다. 원시 텍스트를 사용합니다.")
            return None

def predict_improvement(activity_text, hazard_text, top_k=3, embedding_model="text-embedding-3-small", target_language="Korean"):
    """
    작업활동과 유해위험요인을 기반으로 개선대책을 예측합니다.
    
    Args:
        activity_text: 작업활동 텍스트
        hazard_text: 유해위험요인 텍스트
        top_k: 검색할 유사 예시 개수
        embedding_model: 사용할 임베딩 모델 이름
        target_language: 결과를 출력할 언어 (Korean, Chinese, English)
    
    Returns:
        GPT 모델이 생성한 원시 응답 문자열
    """
    # retriever_pool_df가 세션 상태에 있는지 확인
    if "retriever_pool_df" not in st.session_state:
        st.error("데이터셋이 로드되지 않았습니다. 먼저 임베딩 사전 계산을 실행해주세요.")
        return None
        
    # 쿼리 텍스트 임베딩
    query_text = f"{activity_text} {hazard_text}"
    query_embedding = get_openai_embedding(query_text, model_name=embedding_model)

    # FAISS 검색 (인덱스가 있는지 확인)
    if "phase2_index" not in st.session_state:
        st.error("FAISS 인덱스가 로드되지 않았습니다. 먼저 임베딩 사전 계산을 실행해주세요.")
        return None
        
    distances, idxs = st.session_state.phase2_index.search(query_embedding.reshape(1, -1), top_k)
    retrieved_docs = st.session_state.retriever_pool_df.iloc[idxs[0]]

    # 예시 섹션 구성
    example_section = ""
    for _, row in retrieved_docs.iterrows():
        example_section += (
            "Example:\n"
            f"Input (Activity, Hazard): {row['작업활동 및 내용']} / {row['유해위험요인 및 환경측면 영향']}\n"
            "Output (Improvement Plan, Improved Freq/Sev/T) in JSON:\n"
            "{"
            f"\"개선대책 및 세부관리방안\": \"{row.get('개선대책 및 세부관리방안', 'NA')}\", "
            f"\"개선 후 빈도\": {row.get('개선 후 빈도', 1)}, "
            f"\"개선 후 강도\": {row.get('개선 후 강도', 1)}, "
            f"\"개선 후 T\": {row.get('개선 후 T', 1)}"
            "}\n\n"
        )

    # 최종 프롬프트
    prompt = (
        f"{example_section}"
        "Now here is a new input:\n"
        f"작업활동 및 내용: {activity_text}\n"
        f"유해위험요인 및 환경측면 영향: {hazard_text}\n\n"
        "Please provide the output in JSON format with these keys:\n"
        "{\n"
        "  \"개선대책 및 세부관리방안\": \"...\", \n"
        "  \"개선 후 빈도\": (an integer in [1..5]),\n"
        "  \"개선 후 강도\": (an integer in [1..5]),\n"
        "  \"개선 후 T\": (Improved Frequency * Improved Severity)\n"
        "}\n\n"
        f"Please write the content (개선대책 및 세부관리방안) in {target_language}.\n"
        "Make sure to return only valid JSON.\n"
        "Output:\n"
    )

    # GPT 모델 호출
    gpt_raw = call_gpt(prompt)
    return gpt_raw

def compute_rrr(T_before, T_after):
    """Risk Reduction Rate(RRR) a계산 함수"""
    if T_before == 0:
        return 0.0
    return ((T_before - T_after) / T_before) * 100.0

def main():
    st.set_page_config(
        page_title="위험성 평가 시스템",
        page_icon="🏗️",
        layout="wide"
    )

    # ----- 세션 상태 초기화 -----
    if "index" not in st.session_state:
        st.session_state.index = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "phase2_index" not in st.session_state:
        st.session_state.phase2_index = None
    if "retriever_pool_df" not in st.session_state:
        st.session_state.retriever_pool_df = None

    # 메인 타이틀
    st.title("생성형 AI 기반 위험성 평가 시스템")

    # 데이터셋 선택
    selected_dataset_name = st.selectbox(
        "데이터셋 선택",
        options=list(dataset_options.keys()),
        key="dataset_selector"
    )
    st.write(f"선택된 데이터셋: {selected_dataset_name}")

    # 로고 표시
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image("cau.png", width=200)
        except Exception:
            st.error("중앙대학교 로고 로딩 실패")
    with col2:
        try:
            st.image("doosan.png", width=200)
        except Exception:
            st.error("두산에너빌리티 로고 로딩 실패")

    # API 키 입력
    api_key = st.text_input(
        "OpenAI API 키를 입력하세요:",
        type="password",
        help="API 키는 안전하게 보관됩니다."
    )
    if not api_key:
        st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        return
    openai.api_key = api_key

    # 데이터 불러오기
    with st.spinner('데이터를 불러오는 중...'):
        df = load_data(dataset_options[selected_dataset_name])
    if df is None:
        return

    # train/test 분할
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_df = test_df[['작업활동 및 내용', '유해위험요인 및 환경측면 영향', '빈도', '강도', 'T']]

    # Retriever Pool 구성
    retriever_pool_df = train_df.copy()
    retriever_pool_df['content'] = retriever_pool_df.apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1
    )
    texts = retriever_pool_df['content'].tolist()

    # ----- 탭 구분 -----
    tabs = st.tabs(["임베딩 사전 계산", "사용자 입력 예측", "샘플 예측", "개선대책 생성"])

    # 탭 1) 임베딩 사전 계산
    with tabs[0]:
        st.subheader("임베딩 계산 / 인덱스 구성")

        if st.session_state.index is not None:
            st.success("이미 임베딩 계산 및 인덱스 구성이 완료되었습니다!")
        else:
            if st.button("임베딩 사전 계산", key="run_embedding"):
                with st.spinner('임베딩을 생성하는 중...'):
                    # Phase 1 임베딩 및 인덱스
                    embeddings = embed_texts_with_openai(texts)
                    if not embeddings:
                        st.error("임베딩 생성 실패")
                        return
                    st.session_state.embeddings = np.array(embeddings, dtype='float32')
                    dimension = st.session_state.embeddings.shape[1]
                    faiss_index = faiss.IndexFlatL2(dimension)
                    faiss_index.add(st.session_state.embeddings)
                    st.session_state.index = faiss_index
                    
                    # Phase 2를 위한 데이터 및 인덱스 구성
                    st.session_state.retriever_pool_df = retriever_pool_df
                    
                    # Phase 2 인덱스 초기화 (실제로는 적절한 데이터로 훈련해야 함)
                    phase2_index = faiss.IndexFlatL2(dimension)
                    phase2_index.add(st.session_state.embeddings)
                    st.session_state.phase2_index = phase2_index
                    
                st.success("임베딩 생성 및 인덱스 구성 완료!")
            else:
                st.info("아직 인덱스가 만들어지지 않았습니다. [임베딩 사전 계산]을 눌러주세요.")

    # ----- 공통 설정 (사이드바 등) -----
    with st.sidebar:
        st.header("📊 분석 설정")
        k_similar = st.slider("유사 사례 검색 수", min_value=1, max_value=10, value=5)
        
        # Phase 2 관련 사이드바 설정
        st.header("📋 개선대책 설정")
        target_language = st.selectbox(
            "개선대책 언어 선택", 
            options=language_options,
            index=0
        )

    # 탭 2) 사용자 입력 예측
    with tabs[1]:
        st.subheader("사용자 입력 예측")

        if st.session_state.index is None:
            st.warning("먼저 [임베딩 사전 계산] 탭에서 인덱스를 생성하세요.")
        else:
            with st.form("user_input_form"):
                user_work = st.text_input("작업활동 (사용자 입력):", key="form_user_work")
                user_risk = st.text_input("유해위험요인 (사용자 입력):", key="form_user_risk")
                submitted = st.form_submit_button("사용자 입력으로 예측하기")

            if submitted:
                if not user_work or not user_risk:
                    st.warning("작업활동과 유해위험요인을 모두 입력하세요.")
                else:
                    query_text = f"{user_work} {user_risk}"
                    
                    # 쿼리 임베딩
                    query_embedding = embed_texts_with_openai([query_text])[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    
                    # 유사 문서 검색
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = retriever_pool_df.iloc[indices[0]]

                    # GPT 프롬프트 생성 & 호출
                    prompt = construct_prompt(retrieved_docs, query_text)
                    generated_output = generate_with_gpt4(prompt)

                    st.markdown(f"**사용자 입력 쿼리**: {query_text}")
                    parse_result = parse_gpt_output(generated_output)
                    if parse_result is not None:
                        f_val, i_val, t_val = parse_result
                        st.write(f"GPT 예측 → 빈도: {f_val}, 강도: {i_val}, T: {t_val}")
                    else:
                        st.write(f"GPT 예측(원문): {generated_output}")

    # 탭 3) 샘플 예측
    with tabs[2]:
        st.subheader("샘플 예측 (상위 3개만 표시)")

        if st.session_state.index is None:
            st.warning("먼저 [임베딩 사전 계산] 탭에서 인덱스를 생성하세요.")
        else:
            sample_df = test_df.iloc[:3].copy().reset_index(drop=True)

            for idx, row in sample_df.iterrows():
                st.markdown(f"**샘플 {idx+1}**")
                st.markdown(f"- 작업활동: {row['작업활동 및 내용']}")
                st.markdown(f"- 유해위험요인: {row['유해위험요인 및 환경측면 영향']}")
                st.markdown(f"- 실제 빈도: {row['빈도']}, 실제 강도: {row['강도']}, 실제 T: {row['T']}")

                query_text = f"{row['작업활동 및 내용']} {row['유해위험요인 및 환경측면 영향']}"

                # 쿼리 임베딩
                query_embedding = embed_texts_with_openai([query_text])[0]
                query_embedding_array = np.array([query_embedding], dtype='float32')

                # FAISS 검색
                distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                retrieved_docs = retriever_pool_df.iloc[indices[0]]

                # GPT 호출
                prompt = construct_prompt(retrieved_docs, query_text)
                generated_output = generate_with_gpt4(prompt)

                # GPT 예측 파싱
                parse_result = parse_gpt_output(generated_output)
                if parse_result is not None:
                    f_val, i_val, t_val = parse_result
                    st.write(f"**GPT 예측** → 빈도: {f_val}, 강도: {i_val}, T: {t_val}")
                else:
                    st.write(f"**GPT 예측**: {generated_output}")
                
                st.markdown("---")

            st.markdown("### 예측 완료")
            st.info("상기 표시된 샘플 3개는 실제 데이터셋에서 일부만 발췌한 예시입니다.")
            
    # 탭 4) 개선대책 생성 (Phase 2 추가)
    with tabs[3]:
        st.subheader("개선대책 생성")
        
        if st.session_state.phase2_index is None:
            st.warning("먼저 [임베딩 사전 계산] 탭에서 인덱스를 생성하세요.")
        else:
            st.write("작업활동과 유해위험요인을 입력하여 개선대책을 생성합니다.")
            
            with st.form("improvement_form"):
                p2_user_work = st.text_input("작업활동:", key="p2_user_work")
                p2_user_risk = st.text_input("유해위험요인:", key="p2_user_risk")
                p2_user_freq = st.number_input("개선 전 빈도 (1-5):", min_value=1, max_value=5, value=3)
                p2_user_intensity = st.number_input("개선 전 강도 (1-5):", min_value=1, max_value=5, value=3)
                p2_submitted = st.form_submit_button("개선대책 생성하기")
            
            if p2_submitted:
                if not p2_user_work or not p2_user_risk:
                    st.warning("작업활동과 유해위험요인을 모두 입력하세요.")
                else:
                    # 개선 전 T 값 계산
                    T_before = p2_user_freq * p2_user_intensity
                    
                    with st.spinner('개선대책을 생성하는 중...'):
                        # GPT를 이용한 개선대책 생성
                        gpt_raw = predict_improvement(
                            activity_text=p2_user_work,
                            hazard_text=p2_user_risk,
                            top_k=k_similar,
                            embedding_model="text-embedding-3-small",
                            target_language=target_language
                        )
                        
                        # 결과 파싱
                        parsed_result = parse_gpt_response(gpt_raw)
                        
                    # 결과 표시
                    st.markdown("## 위험성 평가 결과")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 개선 전")
                        st.markdown(f"- **작업활동**: {p2_user_work}")
                        st.markdown(f"- **유해위험요인**: {p2_user_risk}")
                        st.markdown(f"- **빈도**: {p2_user_freq}")
                        st.markdown(f"- **강도**: {p2_user_intensity}")
                        st.markdown(f"- **T**: {T_before}")
                        st.markdown(f"- **등급**: {determine_grade(T_before)}")
                    
                    with col2:
                        st.markdown("### 개선 후 (GPT 예측)")
                        if parsed_result:
                            improvement_plan = parsed_result.get("개선대책 및 세부관리방안", "")
                            improved_freq = parsed_result.get("개선 후 빈도", 1)
                            improved_intensity = parsed_result.get("개선 후 강도", 1)
                            improved_T = parsed_result.get("개선 후 T", 1)
                            
                            # T 값 검증 (빈도 * 강도 = T)
                            if improved_freq * improved_intensity != improved_T:
                                improved_T = improved_freq * improved_intensity
                                st.warning("개선 후 T 값이 빈도 * 강도와 일치하지 않아 다시 계산했습니다.")
                            
                            rrr = compute_rrr(T_before, improved_T)
                            
                            st.markdown(f"- **개선 후 빈도**: {improved_freq}")
                            st.markdown(f"- **개선 후 강도**: {improved_intensity}")
                            st.markdown(f"- **개선 후 T**: {improved_T}")
                            st.markdown(f"- **등급**: {determine_grade(improved_T)}")
                            st.markdown(f"- **위험 감소율**: {rrr:.2f}%")
                        else:
                            st.error("개선대책 생성에 실패했습니다. GPT 응답을 파싱할 수 없습니다.")
