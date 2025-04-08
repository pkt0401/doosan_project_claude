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
from tqdm import tqdm

# ----- 전역 변수 (데이터셋 선택 옵션) -----
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "Marine (토목)": "Marine (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)": "SWRO 전기작업표준 (플랜트)"
}

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

def load_index_file(index_filename="phase1_general_api_updated.index"):
    """미리 계산된 인덱스 파일을 로드하는 함수."""
    try:
        st.info(f"인덱스 파일 '{index_filename}'을 로드하는 중...")
        # FAISS 인덱스 로드
        faiss_index = faiss.read_index(index_filename)
        st.success(f"인덱스 파일 로드 완료: {faiss_index.ntotal}개 벡터 포함")
        return faiss_index
    except Exception as e:
        st.error(f"인덱스 파일 로드 중 오류 발생: {str(e)}")
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

def main():
    st.set_page_config(
        page_title="위험성 평가 시스템",
        page_icon="🏗️",
        layout="wide"
    )

    # ----- 세션 상태 초기화 -----
    # (index, user inputs 등을 보관)
    if "index" not in st.session_state:
        st.session_state.index = None
    if "retriever_pool_df" not in st.session_state:
        st.session_state.retriever_pool_df = None
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False

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

    # Retriever Pool 구성 (세션 상태에 저장)
    if st.session_state.retriever_pool_df is None:
        retriever_pool_df = train_df.copy()
        retriever_pool_df['content'] = retriever_pool_df.apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1
        )
        st.session_state.retriever_pool_df = retriever_pool_df
    
    # 인덱스 자동 로드 (처음 실행시)
    if not st.session_state.index_loaded:
        with st.spinner('인덱스 파일을 자동으로 로드하는 중...'):
            faiss_index = load_index_file("phase1_general_api_updated.index")
            if faiss_index is not None:
                st.session_state.index = faiss_index
                st.session_state.index_loaded = True
                st.success("인덱스 파일이 자동으로 로드되었습니다!")
            else:
                st.error("인덱스 자동 로드 실패")
    
    # ----- 탭 구분 -----
    tabs = st.tabs(["사용자 입력 예측", "샘플 예측"])

    # ----- 공통 설정 (사이드바 등) -----
    with st.sidebar:
        st.header("📊 분석 설정")
        k_similar = st.slider("유사 사례 검색 수", min_value=1, max_value=10, value=5)

    # 탭 1) 사용자 입력 예측
    with tabs[0]:
        st.subheader("사용자 입력 예측")

        if st.session_state.index is None:
            st.warning("인덱스 파일을 로드할 수 없습니다. 파일 경로를 확인하세요.")
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
                    retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]

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

    # 탭 2) 샘플 예측
    with tabs[1]:
        st.subheader("샘플 예측 (상위 3개만 표시)")

        if st.session_state.index is None:
            st.warning("인덱스 파일을 로드할 수 없습니다. 파일 경로를 확인하세요.")
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
                retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]

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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"프로그램 실행 중 오류가 발생했습니다: {str(e)}")
