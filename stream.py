import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
import io  # NEW: for Excel download
from PIL import Image
from sklearn.model_selection import train_test_split

# ------------------------------ 기본 설정 ------------------------------

st.set_page_config(
    page_title="Artificial Intelligence Risk Assessment (KOR)",
    page_icon="🛠️",
    layout="wide",
)

# ------------------------------ 스타일 ------------------------------

st.markdown(
    """
<style>
.main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
.sub-header {font-size: 1.6rem; color: #0D47A1; margin: 1.5rem 0 1rem;}
.metric-container {background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
.result-box {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 5px solid #4CAF50;}
.similar-case {background-color: #f1f8e9; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 4px solid #689f38;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------ 세션 상태 ------------------------------

if "index" not in st.session_state:
    st.session_state.index = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ------------------------------ 유틸 함수 ------------------------------

def determine_grade(value: int) -> str:
    if 16 <= value <= 25:
        return "A"
    elif 10 <= value <= 15:
        return "B"
    elif 5 <= value <= 9:
        return "C"
    elif 3 <= value <= 4:
        return "D"
    elif 1 <= value <= 2:
        return "E"
    return "알 수 없음"


def load_data(name: str) -> pd.DataFrame:
    """엑셀에서 데이터셋을 읽어 DataFrame으로 반환한다."""
    try:
        df = pd.read_excel(f"{name}.xlsx")
        if "삭제 Del" in df.columns:
            df = df.drop(["삭제 Del"], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: "빈도", df.columns[5]: "강도"})
        df["T"] = pd.to_numeric(df.iloc[:, 4]) * pd.to_numeric(df.iloc[:, 5])
        df = df.iloc[:, :7]
        df.rename(
            columns={
                "작업활동 및 내용\nWork & Contents": "작업활동 및 내용",
                "유해위험요인 및 환경측면 영향\nHazard & Risk": "유해위험요인 및 환경측면 영향",
            },
            inplace=True,
        )
        df = df.rename(columns={df.columns[6]: "T"})
        df["등급"] = df["T"].apply(determine_grade)
        return df
    except Exception:
        # 데모용 샘플 데이터
        data = {
            "작업활동 및 내용": ["Shoring Installation", "In and Out of materials"],
            "유해위험요인 및 환경측면 영향": [
                "Ground collapse",
                "Vehicle overturn",
            ],
            "피해형태 및 환경영향": ["Injury", "Damage"],
            "빈도": [3, 3],
            "강도": [2, 3],
        }
        df = pd.DataFrame(data)
        df["T"] = df["빈도"] * df["강도"]
        df["등급"] = df["T"].apply(determine_grade)
        return df


def embed_texts(texts, api_key):
    """주어진 텍스트 목록을 임베딩하여 리스트 반환"""
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        try:
            resp = openai.Embedding.create(model="text-embedding-3-large", input=[str(text)])
            embeddings.append(resp["data"][0]["embedding"])
        except Exception:
            embeddings.append([0] * 1536)
    return embeddings


def gpt_chat(prompt, api_key, model="gpt-4o"):
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256)
    return resp["choices"][0]["message"]["content"].strip()

# ---- Phase 1 prompt builders (Korean only) ----


def prompt_hazard(examples: pd.DataFrame, activity: str) -> str:
    intro = "다음은 건설 현장의 작업활동과 그에 따른 유해위험요인의 예시입니다:\n\n"
    body = "".join(
        [
            f"예시 {i+1}:\n작업활동: {row['작업활동 및 내용']}\n유해위험요인: {row['유해위험요인 및 환경측면 영향']}\n\n"
            for i, row in examples.iterrows()
        ]
    )
    query = f"이제 다음 작업활동에 대한 유해위험요인을 예측해주세요:\n작업활동: {activity}\n유해위험요인: "
    return intro + body + query


def prompt_risk(examples: pd.DataFrame, activity: str, hazard: str) -> str:
    tpl_example = "예시 {i}:\n입력: {inp}\n출력: {{\"빈도\": {f}, \"강도\": {s}, \"T\": {t}}}\n\n"
    body = "".join(
        [
            tpl_example.format(
                i=i + 1,
                inp=f"{row['작업활동 및 내용']} - {row['유해위험요인 및 환경측면 영향']}",
                f=int(row["빈도"]),
                s=int(row["강도"]),
                t=int(row["T"]),
            )
            for i, row in examples.iterrows()
        ]
    )

    query = (
        f"입력: {activity} - {hazard}\n위 입력을 바탕으로 빈도와 강도를 예측하세요. 빈도는 1에서 5 사이의 정수입니다. "
        "강도는 1에서 5 사이의 정수입니다. T는 빈도와 강도를 곱한 값입니다.\n다음 JSON 형식으로 출력하세요:\n"
        "{\"빈도\": 숫자, \"강도\": 숫자, \"T\": 숫자}\n출력:\n"
    )
    return body + query


def parse_risk(output: str):
    m = re.search(r'{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*(\d+)}', output)
    if not m:
        return None
    f, s, t = map(int, m.groups())
    return f, s, t

# ---- Phase 2 prompt & parsing (Korean) ----


def prompt_improvement(examples: pd.DataFrame, activity: str, hazard: str, freq: int, severity: int, t_val: int) -> str:
    body = ""
    added = 0
    for _, row in examples.iterrows():
        if "개선대책" in row and pd.notna(row["개선대책"]):
            imp = row["개선대책"]
            of, os = int(row["빈도"]), int(row["강도"])
            ot = of * os
            body += (
                "Example:\n"
                f"Input (Activity): {row['작업활동 및 내용']}\n"
                f"Input (Hazard): {row['유해위험요인 및 환경측면 영향']}\n"
                f"Input (Original Frequency): {of}\nInput (Original Intensity): {os}\nInput (Original T): {ot}\n"
                "Output (Improvement Plan and Risk Reduction) in JSON:\n"
                "{\n  \"개선대책\": \"%s\",\n  \"개선 후 빈도\": 1,\n  \"개선 후 강도\": 1,\n  \"개선 후 T\": 1,\n  \"T 감소율\": 90.0\n}\n\n" % imp
            )
            added += 1
            if added >= 2:
                break
    query = (
        "다음은 새로운 입력입니다:\n"
        f"입력 (작업활동): {activity}\n입력 (유해위험요인): {hazard}\n입력 (원래 빈도): {freq}\n입력 (원래 강도): {severity}\n입력 (원래 T): {t_val}\n\n"
        "다음 JSON 형식으로 출력을 제공하세요:\n{\n  \"개선대책\": \"항목별 개선대책 리스트\",\n  \"개선 후 빈도\": (1..5),\n  \"개선 후 강도\": (1..5),\n  \"개선 후 T\": (숫자),\n  \"T 감소율\": (퍼센트)\n}\n\n개선대책은 한국어로 작성하십시오. 최소 3개의 구체적인 개선 조치를 번호가 매겨진 목록으로 제공하십시오. 유효한 JSON만 반환하십시오.\n출력:\n"
    )
    return body + query


def parse_improvement(output: str):
    try:
        js = re.search(r'{.*}', output, re.DOTALL)
        if not js:
            return None
        import json
        data = json.loads(js.group())
        keys = ["개선대책", "개선 후 빈도", "개선 후 강도", "개선 후 T", "T 감소율"]
        return {k: data.get(k) for k in keys}
    except Exception:
        return None

# ------------------------------ UI ------------------------------

st.markdown("<div class='main-header'>LLM 기반 위험성 평가 · 개선대책 생성 (통합)</div>", unsafe_allow_html=True)

# 1) API Key & Dataset
api_key = st.text_input("🔑 OpenAI API Key", type="password")

datasets = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)"
}
sel_ds = st.selectbox("데이터셋 선택", list(datasets.keys()))

if st.button("데이터 불러오기 및 인덱스 구축"):
    if not api_key:
        st.warning("API 키를 입력하세요.")
    else:
        with st.spinner("데이터 로드 중..."):
            df = load_data(datasets[sel_ds])
            train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
            pool_df = train_df.copy()
            pool_df["content"] = pool_df.apply(lambda r: " ".join(r.values.astype(str)), axis=1)
            texts = pool_df["content"].tolist()[:10]
            embeds = embed_texts(texts, api_key)
            embeds_np = np.array(embeds, dtype="float32")
            idx = faiss.IndexFlatL2(embeds_np.shape[1])
            idx.add(embeds_np)
            st.session_state.index = idx
            st.session_state.retriever_pool_df = pool_df
        st.success("인덱스가 준비되었습니다!")

st.divider()

# 2) User Input -> All results
work = st.text_input("작업활동 및 내용 입력 후 Enter: ")

if st.button("위험성 평가 + 개선대책 생성"):
    if not api_key:
        st.warning("API 키를 입력하세요.")
    elif st.session_state.index is None:
        st.warning("먼저 데이터셋을 로드하세요.")
    elif not work:
        st.warning("작업활동을 입력하세요.")
    else:
        with st.spinner("모든 결과 생성 중..."):
            # ------- Retrieve examples -------
            q_embed = embed_texts([work], api_key)[0]
            D, I = st.session_state.index.search(np.array([q_embed], dtype="float32"), 3)
            examples = st.session_state.retriever_pool_df.iloc[I[0]].copy()

            # 유사 사례 표시 (카드 형식)
            st.markdown("<div class='sub-header'>유사 사례</div>", unsafe_allow_html=True)
            for i, row in examples.iterrows():
                st.markdown(
                    f"<div class='similar-case'><b>#{i}</b> 작업활동: {row['작업활동 및 내용']}<br>유해위험요인: {row['유해위험요인 및 환경측면 영향']}<br>빈도: {row['빈도']} 강도: {row['강도']} T: {row['T']} 등급: {row['등급']}</div>",
                    unsafe_allow_html=True,
                )

            # 유사 사례 테이블 (NEW)
            st.markdown("##### 유사 사례 상세 (Table)")
            st.dataframe(examples[[
                "작업활동 및 내용", "유해위험요인 및 환경측면 영향", "빈도", "강도", "T", "등급", "개선대책", "개선 후 빈도", "개선 후 강도", "개선 후 T", "T 감소율"
            ]])

            # ------- Phase 1: Hazard -------
            h_prompt = prompt_hazard(examples, work)
            hazard = gpt_chat(h_prompt, api_key)

            r_prompt = prompt_risk(examples, work, hazard)
            risk_out = gpt_chat(r_prompt, api_key)
            risk_vals = parse_risk(risk_out)
            if not risk_vals:
                st.error("빈도/강도 파싱 실패")
                st.write(risk_out)
                st.stop()
            freq, sev, t_val = risk_vals
            grade = determine_grade(t_val)

            # ------- Phase 2: Improvement -------
            i_prompt = prompt_improvement(examples, work, hazard, freq, sev, t_val)
            imp_out = gpt_chat(i_prompt, api_key)
            imp_parsed = parse_improvement(imp_out)
            if not imp_parsed:
                st.error("개선대책 파싱 실패")
                st.write(imp_out)
                st.stop()

        # ----------------- 결과 표시 -----------------
        st.markdown("<div class='sub-header'>예측 결과</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-box'>작업활동: <b>{work}</b><br>예측된 유해위험요인: <b>{hazard}</b></div>",
            unsafe_allow_html=True,
        )

        st.table(
            pd.DataFrame(
                {
                    "항목": ["빈도", "강도", "T", "등급"],
                    "값": [freq, sev, t_val, grade],
                }
            )
        )

        st.markdown("<div class='sub-header'>개선대책 및 위험도 개선</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("##### 개선대책")
            st.markdown(imp_parsed["개선대책"])
        with col2:
            st.markdown("##### 개선 전후 비교")
            imp_freq = imp_parsed.get("개선 후 빈도", 1)
            imp_sev = imp_parsed.get("개선 후 강도", 1)
            imp_t = imp_parsed.get("개선 후 T", imp_freq * imp_sev)
            rrr = imp_parsed.get("T 감소율", (t_val - imp_t) / t_val * 100)
            compare_df = pd.DataFrame(
                {
                    "항목": ["빈도", "강도", "T", "등급"],
                    "개선 전": [freq, sev, t_val, grade],
                    "개선 후": [imp_freq, imp_sev, imp_t, determine_grade(imp_t)],
                }
            )
            st.table(compare_df)
            st.metric("T 감소율", f"{rrr:.2f}%")

        st.progress(imp_t / 25)

        # ----------------- 결과 Excel 다운로드 (NEW) -----------------
        st.markdown("<div class='sub-header'>📊 결과 다운로드</div>", unsafe_allow_html=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # 시트 1: 예측 요약
            summary_df = pd.DataFrame({
                "항목": ["작업활동", "예상 유해위험요인", "빈도", "강도", "T", "등급", "개선대책", "개선 후 빈도", "개선 후 강도", "개선 후 T", "T 감소율"],
                "값": [work, hazard, freq, sev, t_val, grade, imp_parsed["개선대책"], imp_freq, imp_sev, imp_t, rrr],
            })
            summary_df.to_csv(writer, index=False, sheet_name="Summary")
            # 시트 2: 유사 사례
            examples.to_csv(writer, index=False, sheet_name="Similar Cases")
        output.seek(0)


        
        st.download_button(
            label="📥 결과 Excel 다운로드",
            data=output.getvalue(),
            file_name="risk_assessment_result.csv",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ------------------------------ 로고 ------------------------------

st.divider()
cols = st.columns(2)
for img_path, w, col in [("cau.png", 150, cols[0]), ("doosan.png", 180, cols[1])]:
    if os.path.exists(img_path):
        col.image(Image.open(img_path), width=w)
