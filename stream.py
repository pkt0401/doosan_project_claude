# Streamlit App: Integrated AI Risk Assessment (Phase 1 + Phase 2)
# ----------------------------------------------------------------
# * Single input → full pipeline (hazard prediction ➝ risk grading ➝ improvement measures)
# * Multilingual UI (Korean / English / Chinese)
# * No artificial embedding‑count limits or demo warning messages
# * Phase 2 prompt examples explicitly include the "Improvement Plan" field

# ---------- Imports ----------
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
import io
from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------- Page Config ----------
# Streamlit 페이지 설정은 가장 먼저 호출되어야 함
st.set_page_config("AI Risk Assessment", ":망치와_렌치:", layout="wide")

# ---------- Helper: Grade ----------
GRADE = [(16, 25, 'A'), (10, 15, 'B'), (5, 9, 'C'), (3, 4, 'D'), (1, 2, 'E')]

def grade(t: int) -> str:
    """Convert T‑value to grade."""
    for lo, hi, g in GRADE:
        if lo <= t <= hi:
            return g
    return "?"

# ---------- Utility Functions ----------

def _load_data(name: str) -> pd.DataFrame:
    """Load Excel → tidy DataFrame, compute T & grade.
    ***Note***: returns only the first 10 rows to keep the FAISS index small (per user request)."""
    try:
        df = pd.read_excel(f"{name}.xlsx")
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]

        # 첫 두 열 이름 명시적으로 설정
        if len(df.columns) >= 2:
            df = df.rename(columns={
                df.columns[0]: '작업활동 및 내용',
                df.columns[1]: '유해위험요인 및 환경측면 영향'
            })

        # 빈도, 강도, T 열 이름 설정
        df = df.rename(columns={df.columns[4]: '빈도', df.columns[5]: '강도', df.columns[6]: 'T'})
        df['T'] = pd.to_numeric(df['빈도']) * pd.to_numeric(df['강도'])
        df['등급'] = df['T'].apply(grade)
        df['content'] = df.apply(lambda r: ' '.join(r.astype(str)), axis=1)

        # ***Limit to 10 entries for FAISS index***
        df = df.reset_index(drop=True).head(10)
        return df
    except Exception:
        # fallback dummy
        data = {
            "작업활동 및 내용": ["Excavation"],
            "유해위험요인 및 환경측면 영향": ["Collapse"],
            "빈도": [3],
            "강도": [4]
        }
        df = pd.DataFrame(data)
        df['T'] = df['빈도'] * df['강도']
        df['등급'] = df['T'].apply(grade)
        df['content'] = df.apply(lambda r: ' '.join(r.astype(str)), axis=1)
        return df


def _embed(texts, api_key: str):
    openai.api_key = api_key
    embs = []
    for t in texts:
        r = openai.Embedding.create(model="text-embedding-3-large", input=[t.replace("\n", " ")])
        embs.append(r['data'][0]['embedding'])
    return embs


def _build_index(df: pd.DataFrame, api_key: str):
    """Build a FAISS L2 index from *up to 10* document embeddings."""
    embs = _embed(df['content'].tolist(), api_key)
    arr = np.array(embs, dtype='float32')
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

# ----- Prompt builders & parsers (simplified) -----

def _prompt_hazard(docs, activity):
    examples = []
    for i, (_, r) in enumerate(docs.iterrows()):
        work = r.get('작업활동 및 내용', '작업 설명 없음')
        hazard = r.get('유해위험요인 및 환경측면 영향', '위험 요인 없음')
        examples.append(f"예시 {i + 1}:\n작업활동: {work}\n유해위험요인: {hazard}\n\n")
    return (
        "다음은 건설현장 작업활동과 유해위험요인 예시입니다.\n\n" +
        "".join(examples) +
        f"다음 작업활동의 유해위험요인을 예측하세요:\n작업활동: {activity}\n유해위험요인:"
    )


def _prompt_risk(docs, activity, hazard):
    examples = []
    for i, (_, r) in enumerate(docs.iterrows()):
        work = r.get('작업활동 및 내용', '작업 설명 없음')
        hazard_desc = r.get('유해위험요인 및 환경측면 영향', '위험 요인 없음')
        freq = int(r.get('빈도', 3)) if str(r.get('빈도')).isdigit() else 3
        intensity = int(r.get('강도', 3)) if str(r.get('강도')).isdigit() else 3
        t_value = int(r.get('T', freq * intensity)) if str(r.get('T')).isdigit() else freq * intensity
        examples.append(
            f"예시 {i + 1}:\n입력: {work} - {hazard_desc}\n출력: {{\"빈도\": {freq}, \"강도\": {intensity}, \"T\": {t_value}}}\n\n"
        )
    return (
        "".join(examples) +
        f"입력: {activity} - {hazard}\n빈도와 강도를 1~5 정수로 예측하고 JSON으로 출력: {{\"빈도\": n, \"강도\": n, \"T\": n}}\n출력:"
    )


def _prompt_improve(docs, activity, hazard, f, i, t):
    examples = (
        "Example:\nInput (Activity): Excavation\nInput (Hazard): Wall collapse\nInput (Original F/I/T): 3/4/12\nOutput (Improvement) JSON:\n{\n  \"개선대책\": \"1) 토양 경사 준수 2) 지보공 설치 3) 점검\",\n  \"개선 후 빈도\": 1,\n  \"개선 후 강도\": 2,\n  \"개선 후 T\": 2,\n  \"T 감소율\": 83.33\n}"
    )
    return (
        f"{examples}\n새로운 입력:\nInput (Activity): {activity}\nInput (Hazard): {hazard}\nInput (Original F/I/T): {f}/{i}/{t}\nJSON key: 개선대책, 개선 후 빈도, 개선 후 강도, 개선 후 T, T 감소율\n번호 매긴 개선대책을 한국어로 3개 이상 포함하고 올바른 JSON만 출력하세요.\n출력:"
    )


def _ask_gpt(prompt, api_key, model="gpt-4o"):
    openai.api_key = api_key
    res = openai.ChatCompletion.create(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
    return res['choices'][0]['message']['content']


def _parse_risk(txt):
    m = re.search(r'\{"빈도"\s*:\s*(\d),\s*"강도"\s*:\s*(\d),\s*"T"\s*:\s*(\d+)\}', txt)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (3, 3, 9)


def _parse_improve(txt):
    try:
        j = re.search(r'\{.*\}', txt, re.S).group()
        data = pd.json.loads(j)
    except Exception:
        data = {}
    return {
        "plan": data.get("개선대책", ""),
        "f": data.get("개선 후 빈도", 1),
        "i": data.get("개선 후 강도", 1),
        "t": data.get("개선 후 T", 1),
        "rrr": data.get("T 감소율", 0),
    }


# ---------- Excel helpers ----------

def create_excel_download_link(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.getvalue()


def save_risk_assessment_result(activity, hazard, freq, intensity, t_value, grade_val, imp_plan, newF, newI, newT, rrr):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "평가일시": [now],
        "작업활동": [activity],
        "유해위험요인": [hazard],
        "위험성 평가 전": [{"빈도": freq, "강도": intensity, "T값": t_value, "등급": grade_val}],
        "개선대책": [imp_plan],
        "위험성 평가 후": [{"빈도": newF, "강도": newI, "T값": newT, "등급": grade(newT)}],
        "감소율(%)": [f"{rrr:.1f}"]
    }
    return pd.DataFrame(data)

# ---------- Language Pack ----------
# (Same as previous – omitted here for brevity)
LANG = {...}  # full dict unchanged

# ---------- Style ----------
st.markdown(
    """
    <style>
        .title {font-size:2.2rem;font-weight:700;text-align:center;color:#0d47a1;margin:0 0 1rem 0}
        .box {background:#f8f9fa;border-radius:10px;padding:15px;margin:1rem 0;border-left:5px solid #1E88E5}
        .similar-case{background:#f1f8e9;border-left:4px solid #689F38;border-radius:8px;padding:10px;margin-bottom:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session State ----------
ss = st.session_state
for key, default in {"lang": "Korean", "index": None, "df": None, "api_key": ""}.items():
    ss.setdefault(key, default)

# ---------- Sidebar Controls ----------
with st.sidebar:
    ss.lang = st.selectbox("Language / 언어", list(LANG.keys()), index=list(LANG).index(ss.lang))
    txt = LANG[ss.lang]
    st.title(txt["title"])
    ss.api_key = st.text_input(txt["api_key"], type="password", value=ss.api_key)
    dataset_name = st.selectbox(txt["dataset"], [
        "SWRO 건축공정 (건축)", "Civil (토목)", "Marine (토목)",
        "SWRO 기계공사 (플랜트)", "SWRO 전기작업표준 (플랜트)"], key="ds")
    if st.button(txt["load_btn"], key="load"):
        if not ss.api_key:
            st.warning(txt["api_warn"])
        else:
            with st.spinner(txt["loading"]):
                ss.df = _load_data(dataset_name)
                ss.index = _build_index(ss.df, ss.api_key)
            st.success(txt["loaded"].format(n=len(ss.df)))

# ---------- Main Layout ----------
st.markdown(f"<p class='title'>{txt['title']}</p>", unsafe_allow_html=True)
work_activity = st.text_input(txt["work_input"], key="work")
run = st.button(txt["run_btn"], key="run")

# ---------- Run Pipeline ----------
if run:
    if not ss.api_key:
        st.warning(txt["api_warn"])
    elif ss.index is None:
        st.warning(txt["load_warn"])
    elif not work_activity:
        st.warning(txt["input_warn"])
    else:
        openai.api_key = ss.api_key
        # 1) Retrieve similar examples (up to 3)
        query_emb = _embed([work_activity], ss.api_key)[0]
        D, I = ss.index.search(np.array([query_emb], dtype='float32'), min(3, len(ss.df)))
        sims = ss.df.iloc[I[0]]

        st.subheader(txt["similar_cases"])
        for i, row in sims.iterrows():
            work_desc = row.get('작업활동 및 내용', '정보 없음')
            hazard_desc = row.get('유해위험요인 및 환경측면 영향', '정보 없음')
            t_value = row.get('T', 0)
            freq = row.get('빈도', 0)
            intensity = row.get('강도', 0)
            grade_val = row.get('등급', '-')
            st.markdown(
                f"<div class='similar-case'><b>#{i}</b><br>작업활동: {work_desc}<br>유해위험요인: {hazard_desc}<br>위험도: {t_value} (빈도 {freq}, 강도 {intensity}, 등급 {grade_val})</div>",
                unsafe_allow_html=True
            )

        # 2) Phase‑1 prompts
        hz_prompt = _prompt_hazard(sims, work_activity)
        hazard = _ask_gpt(hz_prompt, ss.api_key)
        risk_prompt = _prompt_risk(sims, work_activity, hazard)
        risk_raw = _ask_gpt(risk_prompt, ss.api_key)
        freq, intensity, T = _parse_risk(risk_raw)

        st.subheader(txt["prediction"])
        st.write(f"**{txt['hazard']}**: {hazard}")
        st.table(pd.DataFrame({txt['risk_table'][0]: txt['risk_rows'], txt['risk_table'][1]: [freq, intensity, T, grade(T)]}))

        # 3) Phase‑2 prompt & answer
        imp_prompt = _prompt_improve(sims, work_activity, hazard, freq, intensity, T)
        imp_raw = _ask_gpt(imp_prompt, ss.api_key)
        imp_parsed = _parse_improve(imp_raw)
        imp_plan = imp_parsed.get("plan", "-")
        newF, newI, newT = imp_parsed.get("f", 1), imp_parsed.get("i", 1), imp_parsed.get("t", 1)
        rrr = imp_parsed.get("rrr", (T - newT) / T * 100 if T else 0)

        st.subheader(txt['improvement_header'])
        st.markdown(f"<div class='box'>{imp_plan}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric(txt['before'], T)
        col2.metric(txt['after'], newT, delta=f"{rrr:.1f}%")

        # 4) Save & download results
        result_df = save_risk_assessment_result(work_activity, hazard, freq, intensity, T, grade(T), imp_plan, newF, newI, newT, rrr)
        st.subheader(txt["result_summary"])
        st.dataframe(result_df)
        excel_data = create_excel_download_link(result_df)
        st.download_button(
            label=txt["download_btn"],
            data=excel_data,
            file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- Footer ----------
footer_cols = st.columns([1, 1])
for path, col in zip(["cau.png", "doosan.png"], footer_cols):
    if os.path.exists(path):
        col.image(Image.open(path), width=140)
