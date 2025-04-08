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

def check_files_exist():
    """필요한 파일들이 존재하는지 확인하는 함수"""
    missing_files = []
    for dataset_name in dataset_options.values():
        if not os.path.exists(f"{dataset_name}.xlsx"):
            missing_files.append(f"{dataset_name}.xlsx")
    
    if not os.path.exists("phase1_general_api_updated.index"):
        missing_files.append("phase1_general_api_updated.index")
    
    if not os.path.exists("cau.png"):
        missing_files.append("cau.png")
    
    if not os.path.exists("doosan.png"):
        missing_files.append("doosan.png")
    
    if missing_files:
        st.error(f"다음 파일을 찾을 수 없습니다: {', '.join(missing_files)}")
        st.info("모든 필요 파일이 현재 디렉토리에 있는지 확인하세요.")
        return False
    return True

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
        file_path = f"{selected_dataset_name}.xlsx"
        st.info(f"데이터 파일 '{file_path}'을 로드하는 중...")
        df = pd.read_excel(file_path)

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

        st.success(f"데이터 로드 완료: {len(df)}개 행 로드됨")
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        st.write(f"시도한 파일 경로: {selected_dataset_name}.xlsx")
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

def validate_api_key(api_key):
    """API 키 유효성 검증"""
    try:
        openai.api_key = api_key
        # 간단한 API 호출로 검증
        openai.Embedding.create(
            model="text-embedding-3-small",
            input=["API 키 검증용 텍스트"]
        )
        return True
    except Exception as e:
        st.error(f"API 키 검증 실패: {str(e)}")
        return False

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

# 인덱스 자동 로드 (처음 실행시)
if not st.session_state.index_loaded:
    with st.spinner('인덱스 파일을 자동으로 로드하는 중...'):
        try:
            faiss_index = load_index_file("phase1_general_api_updated.index")
            if faiss_index is not None:
                st.session_state.index = faiss_index
                st.session_state.index_loaded = True
                st.success(f"인덱스 파일이 자동으로 로드되었습니다! 차원: {faiss_index.d}")
                # 임베딩 모델을 인덱스 차원에 맞게 설정
                if faiss_index.d == 1536:
                    st.session_state.embedding_model = "text-embedding-3-large"
                elif faiss_index.d == 768:
                    st.session_state.embedding_model = "text-embedding-ada-002"
                elif faiss_index.d == 1024:
                    st.session_state.embedding_model = "text-embedding-3-small"
                else:
                    st.warning(f"인덱스 차원({faiss_index.d})에 맞는 임베딩 모델을 식별할 수 없습니다. 기본값 사용.")
                    st.session_state.embedding_model = "text-embedding-3-large"
            else:
                st.error("인덱스 자동 로드 실패")
                return
        except Exception as e:
            st.error(f"인덱스 로드 중 오류 발생: {str(e)}")
            return

# embed_texts_with_openai 함수 수정
def embed_texts_with_openai(texts, model=None):
    """OpenAI 임베딩 API로 텍스트 리스트를 임베딩."""
    # 세션 상태에서 모델 가져오기 (없으면 기본값)
    if model is None:
        model = st.session_state.get('embedding_model', "text-embedding-3-large")
        
    st.write(f"사용 중인 임베딩 모델: {model}")
    
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)

    for idx, text in enumerate(tqdm(texts, desc="임베딩 진행 중", unit="개")):
        try:
            text = text.replace("\n", " ")
            response = openai.Embedding.create(model=model, input=[text])
            embedding = response["data"][0]["embedding"]
            embeddings.append(embedding)
            
            # 첫 번째 임베딩 후 차원 확인 및 출력
            if idx == 0:
                st.write(f"생성된 임베딩 차원: {len(embedding)}")
                if hasattr(st.session_state, 'index') and st.session_state.index is not None:
                    if len(embedding) != st.session_state.index.d:
                        st.error(f"임베딩 차원({len(embedding)})이 인덱스 차원({st.session_state.index.d})과 일치하지 않습니다!")
        except Exception as e:
            st.error(f"텍스트 임베딩 중 오류 발생: {str(e)}")
            embeddings.append([0]*st.session_state.index.d)  # 인덱스 차원에 맞게 조정
        
        progress_bar.progress((idx + 1) / total)
    
    return embeddings

def construct_prompt(retrieved_docs, query_text):
    """검색된 문서들로부터 예시를 구성해 GPT 프롬프트 생성."""
    retrieved_examples = []
    try:
        for _, doc in retrieved_docs.iterrows():
            content_parts = doc['content'].split()
            example_input = ' '.join(content_parts[:-6])
            frequency = int(content_parts[-4])
            intensity = int(content_parts[-3])
            T_value = frequency * intensity
            example_output = f'{{"빈도": {frequency}, "강도": {intensity}, "T": {T_value}}}'
            retrieved_examples.append((example_input, example_output))
    except Exception as e:
        st.error(f"프롬프트 구성 중 오류 발생: {str(e)}")
        return None
    
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
    if not gpt_output:
        return None
        
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

    # 파일 존재 확인
    if not check_files_exist():
        st.stop()

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
        except Exception as e:
            st.error(f"중앙대학교 로고 로딩 실패: {str(e)}")
    with col2:
        try:
            st.image("doosan.png", width=200)
        except Exception as e:
            st.error(f"두산에너빌리티 로고 로딩 실패: {str(e)}")

    # API 키 입력
    api_key = st.text_input(
        "OpenAI API 키를 입력하세요:",
        type="password",
        help="API 키는 안전하게 보관됩니다."
    )
    if not api_key:
        st.warning("계속하려면 OpenAI API 키를 입력하세요.")
        return
    
    # API 키 검증
    if not validate_api_key(api_key):
        st.warning("유효하지 않은 API 키입니다. 올바른 API 키를 입력하세요.")
        return
    
    openai.api_key = api_key

    # 데이터 불러오기
    with st.spinner('데이터를 불러오는 중...'):
        df = load_data(dataset_options[selected_dataset_name])
    if df is None:
        st.error("데이터 로드에 실패했습니다. 파일 경로를 확인하세요.")
        return

    # train/test 분할
    try:
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        test_df = test_df[['작업활동 및 내용', '유해위험요인 및 환경측면 영향', '빈도', '강도', 'T']]
    except Exception as e:
        st.error(f"데이터 분할 중 오류 발생: {str(e)}")
        return

    # Retriever Pool 구성 (세션 상태에 저장)
    if st.session_state.retriever_pool_df is None:
        try:
            retriever_pool_df = train_df.copy()
            retriever_pool_df['content'] = retriever_pool_df.apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            )
            st.session_state.retriever_pool_df = retriever_pool_df
        except Exception as e:
            st.error(f"검색 풀 구성 중 오류 발생: {str(e)}")
            return
    
    # 인덱스 자동 로드 (처음 실행시)
    if not st.session_state.index_loaded:
        with st.spinner('인덱스 파일을 자동으로 로드하는 중...'):
            try:
                faiss_index = load_index_file("phase1_general_api_updated.index")
                if faiss_index is not None:
                    st.session_state.index = faiss_index
                    st.session_state.index_loaded = True
                    st.success("인덱스 파일이 자동으로 로드되었습니다!")
                else:
                    st.error("인덱스 자동 로드 실패")
                    return
            except Exception as e:
                st.error(f"인덱스 로드 중 오류 발생: {str(e)}")
                return
    
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
                    
                    try:
                        # 쿼리 임베딩
                        with st.spinner('쿼리 임베딩 생성 중...'):
                            query_embedding = embed_texts_with_openai([query_text])[0]
                            query_embedding_array = np.array([query_embedding], dtype='float32')
                        
                        # 유사 문서 검색
                        with st.spinner('유사 사례 검색 중...'):
                            distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                            
                            # 인덱스 범위 검증
                            valid_indices = [idx for idx in indices[0] if idx < len(st.session_state.retriever_pool_df)]
                            if len(valid_indices) == 0:
                                st.error("인덱스와 데이터프레임 간 불일치가 발생했습니다.")
                                return
                                
                            retrieved_docs = st.session_state.retriever_pool_df.iloc[valid_indices]

                        # GPT 프롬프트 생성 & 호출
                        with st.spinner('GPT 모델 호출 중...'):
                            prompt = construct_prompt(retrieved_docs, query_text)
                            if not prompt:
                                st.error("프롬프트 생성에 실패했습니다.")
                                return
                                
                            generated_output = generate_with_gpt4(prompt)

                        st.markdown(f"**사용자 입력 쿼리**: {query_text}")
                        parse_result = parse_gpt_output(generated_output)
                        if parse_result is not None:
                            f_val, i_val, t_val = parse_result
                            grade = determine_grade(t_val)
                            st.write(f"GPT 예측 → 빈도: {f_val}, 강도: {i_val}, T: {t_val}, 등급: {grade}")
                        else:
                            st.write(f"GPT 예측(원문): {generated_output}")
                    except Exception as e:
                        st.error(f"예측 과정에서 오류 발생: {str(e)}")

    # 탭 2) 샘플 예측
    # 탭 2) 샘플 예측
        with tabs[1]:
            st.subheader("샘플 예측 (상위 3개만 표시)")
        
            if st.session_state.index is None:
                st.warning("인덱스 파일을 로드할 수 없습니다. 파일 경로를 확인하세요.")
            else:
                # 디버깅 정보 출력
                st.write(f"인덱스 벡터 수: {st.session_state.index.ntotal}")
                st.write(f"데이터프레임 행 수: {len(st.session_state.retriever_pool_df)}")
                st.write(f"test_df 행 수: {len(test_df)}")
                
                try:
                    sample_df = test_df.iloc[:3].copy().reset_index(drop=True)
                    st.write(f"샘플 데이터프레임 행 수: {len(sample_df)}")
        
                    for idx, row in sample_df.iterrows():
                        st.markdown(f"**샘플 {idx+1}**")
                        st.markdown(f"- 작업활동: {row['작업활동 및 내용']}")
                        st.markdown(f"- 유해위험요인: {row['유해위험요인 및 환경측면 영향']}")
                        st.markdown(f"- 실제 빈도: {row['빈도']}, 실제 강도: {row['강도']}, 실제 T: {row['T']}")
        
                        query_text = f"{row['작업활동 및 내용']} {row['유해위험요인 및 환경측면 영향']}"
                        st.write(f"쿼리 텍스트: {query_text}")
        
                        try:
                            # 쿼리 임베딩
                            st.write("임베딩 생성 중...")
                            query_embedding = embed_texts_with_openai([query_text])[0]
                            query_embedding_array = np.array([query_embedding], dtype='float32')
                            st.write("임베딩 생성 완료")
        
                            # FAISS 검색
                            st.write("유사 문서 검색 중...")
                            distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                            st.write(f"검색된 인덱스: {indices[0]}")
                            st.write(f"검색된 거리: {distances[0]}")
                            
                            # 인덱스 범위 검증
                            st.write(f"최대 인덱스 값: {max(indices[0]) if len(indices[0]) > 0 else 'No indices'}")
                            st.write(f"데이터프레임 크기: {len(st.session_state.retriever_pool_df)}")
                            
                            valid_indices = [i for i in indices[0] if i < len(st.session_state.retriever_pool_df)]
                            st.write(f"유효한 인덱스 수: {len(valid_indices)}")
                            
                            if len(valid_indices) == 0:
                                st.error(f"샘플 {idx+1}: 인덱스와 데이터프레임 간 불일치가 발생했습니다.")
                                st.error("검색된 인덱스가 모두 데이터프레임 범위를 벗어납니다.")
                                continue
                                
                            retrieved_docs = st.session_state.retriever_pool_df.iloc[valid_indices]
                            st.write(f"검색된 문서 수: {len(retrieved_docs)}")
                            
                            # 첫 번째 검색 문서 내용 샘플 표시
                            if len(retrieved_docs) > 0:
                                st.write("첫 번째 검색 문서 내용 샘플:")
                                st.write(retrieved_docs.iloc[0]['content'][:200] + "...")
        
                            # GPT 호출
                            st.write("프롬프트 생성 중...")
                            prompt = construct_prompt(retrieved_docs, query_text)
                            if not prompt:
                                st.error(f"샘플 {idx+1}: 프롬프트 생성에 실패했습니다.")
                                continue
                            
                            st.write("GPT 모델 호출 중...")
                            generated_output = generate_with_gpt4(prompt)
                            st.write(f"GPT 원본 출력: {generated_output}")
        
                            # GPT 예측 파싱
                            parse_result = parse_gpt_output(generated_output)
                            if parse_result is not None:
                                f_val, i_val, t_val = parse_result
                                grade = determine_grade(t_val)
                                st.write(f"**GPT 예측** → 빈도: {f_val}, 강도: {i_val}, T: {t_val}, 등급: {grade}")
                            else:
                                st.write(f"**GPT 예측(파싱 실패)**: {generated_output}")
                        
                        except Exception as e:
                            st.error(f"샘플 {idx+1} 처리 중 오류 발생: {str(e)}")
                            import traceback
                            st.error(f"스택 트레이스: {traceback.format_exc()}")
                        
                        st.markdown("---")
        
                    st.markdown("### 예측 완료")
                    st.info("상기 표시된 샘플 3개는 실제 데이터셋에서 일부만 발췌한 예시입니다.")
                
                except Exception as e:
                    st.error(f"샘플 예측 과정에서 오류 발생: {str(e)}")
                    import traceback
                    st.error(f"스택 트레이스: {traceback.format_exc()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"프로그램 실행 중 오류가 발생했습니다: {str(e)}")
