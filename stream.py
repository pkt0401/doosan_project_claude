import os
import re
import json
import time
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. LANGUAGE & UI TEXT MAPPINGS
# -----------------------------------------------------------------------------
FLAG = {"Korean": "🇰🇷", "English": "🇺🇸", "Chinese": "🇨🇳"}

SYSTEM_TEXT = {
    # (기존 딕셔너리에서 Overview, Labels 등 최소한만 유지)
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "input_header": "작업활동 및 내용 입력",
        "api_key_label": "OpenAI API Key 입력",
        "run_btn": "위험성 평가 실행",
        "embedding_msg": "임베딩 진행 중... (문서 {cur}/{total})",
        "similar_cases": "유사 사례",
        "result_header": "AI Risk Assessment 결과",
        "improvement_header": "개선대책 및 위험도 개선",
        "export_btn": "Excel Export",
        "hazard_label": "예측된 유해위험요인",
        "grade": "위험등급",
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "input_header": "Work activity",
        "api_key_label": "OpenAI API Key",
        "run_btn": "Run Assessment",
        "embedding_msg": "Embedding documents... ({cur}/{total})",
        "similar_cases": "Similar cases",
        "result_header": "AI Risk Assessment Results",
        "improvement_header": "Improvement Plan & Risk Mitigation",
        "export_btn": "Excel Export",
        "hazard_label": "Predicted hazard",
        "grade": "Risk grade",
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "input_header": "工作活动",
        "api_key_label": "OpenAI API 密钥",
        "run_btn": "开始评估",
        "embedding_msg": "正在生成嵌入... ({cur}/{total})",
        "similar_cases": "相似案例",
        "result_header": "AI 风险评估结果",
        "improvement_header": "改进措施与风险降低",
        "export_btn": "Excel 导出",
        "hazard_label": "预测的危害",
        "grade": "风险等级",
    },
}

# -----------------------------------------------------------------------------
# 2. PAGE CONFIG & GLOBAL SESSION STATE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Risk Assessment", page_icon="🛠️", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "Korean"
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "retriever_df" not in st.session_state:
    st.session_state.retriever_df = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -----------------------------------------------------------------------------
# 3. TOP BAR  (Language selector + Flags)
# -----------------------------------------------------------------------------

# We draw the top bar using columns so that the language dropdown sticks to the right.
bar_col1, bar_col2 = st.columns([8, 1])
with bar_col2:
    choice = st.selectbox("Language", options=list(SYSTEM_TEXT.keys()),
                          format_func=lambda x: f"{FLAG[x]}  {x}",
                          index=list(SYSTEM_TEXT.keys()).index(st.session_state.lang))
    st.session_state.lang = choice
TXT = SYSTEM_TEXT[st.session_state.lang]

# -----------------------------------------------------------------------------
# 4. HEADER
# -----------------------------------------------------------------------------
st.markdown(f"<h1 style='text-align:center;color:#1565C0;margin-bottom:0.2em'>{TXT['title']}</h1>", unsafe_allow_html=True)

# Logos (좌측: 두산 / 우측: 중앙대)
logo_col1, _, logo_col2 = st.columns([1, 8, 1])
with logo_col1:
    if os.path.exists("doosan.png"):
        st.image("doosan.png", width=120)
with logo_col2:
    if os.path.exists("cau.png"):
        st.image("cau.png", width=90)

# -----------------------------------------------------------------------------
# 5. SIDEBAR  (Input controls)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header(FLAG[st.session_state.lang] + "  " + TXT["input_header"])
    api_key = st.text_input(TXT["api_key_label"], type="password")
    user_activity = st.text_area("", height=120)
    run = st.button(TXT["run_btn"], use_container_width=True)

# -----------------------------------------------------------------------------
# 6. DATA LOADING & EMBEDDING UTILS
# -----------------------------------------------------------------------------

def determine_grade(t: int, lang: str) -> str:
    if 16 <= t <= 25:
        return "A"
    if 10 <= t <= 15:
        return "B"
    if 5 <= t <= 9:
        return "C"
    if 3 <= t <= 4:
        return "D"
    if 1 <= t <= 2:
        return "E"
    return "Unknown" if lang != "Korean" else "알 수 없음"

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load Excel or fallback sample."""
    try:
        df = pd.read_excel("Civil (토목).xlsx")  # 실제 파일명
        if "삭제 Del" in df.columns:
            df = df.drop(["삭제 Del"], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: "빈도", df.columns[5]: "강도", df.columns[6]: "T"})
        df["T"] = pd.to_numeric(df["빈도"]) * pd.to_numeric(df["강도"])
        df["등급"] = df["T"].apply(lambda x: determine_grade(int(x), "Korean"))
        return df
    except Exception:
        # Minimal sample – production should replace with real file.
        _samp = {
            "작업활동 및 내용": ["Lifting operation", "Excavation work"],
            "유해위험요인 및 환경측면 영향": ["Material fall", "Wall collapse"],
            "피해형태 및 환경영향": ["Injury", "Injury"],
            "빈도": [3, 4],
            "강도": [4, 4],
        }
        df = pd.DataFrame(_samp)
        df["T"] = df["빈도"] * df["강도"]
        df["등급"] = df["T"].apply(lambda x: determine_grade(int(x), "Korean"))
        return df

def build_embeddings(df: pd.DataFrame, api_key: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    """Embed full dataset and return a FAISS index."""
    openai.api_key = api_key
    contents = df.apply(lambda r: " ".join(r.astype(str)), axis=1).tolist()
    embeddings: List[List[float]] = []
    pb = st.progress(0, text=TXT["embedding_msg"].format(cur=0, total=len(contents)))
    for idx, text in enumerate(contents, 1):
        try:
            resp = openai.Embedding.create(model="text-embedding-3-large", input=[text])
            embeddings.append(resp["data"][0]["embedding"])
        except Exception as e:
            st.error(f"Embedding error @ row {idx}: {e}")
            embeddings.append([0.0] * 1536)
        pb.progress(idx / len(contents), text=TXT["embedding_msg"].format(cur=idx, total=len(contents)))
    emb_arr = np.array(embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(emb_arr.shape[1])
    idx.add(emb_arr)
    return idx, emb_arr

# -----------------------------------------------------------------------------
# 7. GPT HELPERS
# -----------------------------------------------------------------------------

def gpt_chat(prompt: str, api_key: str, lang: str, max_tokens: int = 256) -> str:
    openai.api_key = api_key
    system = {
        "Korean": "당신은 건설 안전 전문가입니다. 모든 답변은 한국어로, 공학적 수치를 포함한 구체적 지침을 제공합니다.",
        "English": "You are a construction safety expert. Answer in English with engineering‑level, quantitative guidance.",
        "Chinese": "你是一名建筑安全专家。请用中文回答，并给出具有工程量化指标的具体措施。",
    }[lang]
    resp = openai.ChatCompletion.create(
        model="gpt-4o", temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp["choices"][0]["message"]["content"].strip()

# -----------------------------------------------------------------------------
# 8. PROMPT BUILDERS  (Hazard + Risk + Improvement)
# -----------------------------------------------------------------------------

def prompt_hazard(examples: pd.DataFrame, activity: str, lang: str) -> str:
    intro = {
        "Korean": "다음은 작업활동과 유해위험요인 예시입니다:\n\n",
        "English": "Examples of work activities and hazards:\n\n",
        "Chinese": "以下是工作活动及危害示例：\n\n",
    }[lang]
    fmt = {
        "Korean": "예시 {i}: 작업활동: {a}\n유해위험요인: {h}\n\n",
        "English": "Example {i}: Activity: {a}\nHazard: {h}\n\n",
        "Chinese": "示例 {i}: 工作活动: {a}\n危害: {h}\n\n",
    }[lang]
    q = {
        "Korean": "다음 작업활동의 유해위험요인을 구체적으로 예측하십시오:\n작업활동: {act}\n유해위험요인:",
        "English": "Predict the specific hazard for the following activity:\nActivity: {act}\nHazard:",
        "Chinese": "请预测以下工作活动的具体危害：\n工作活动: {act}\n危害:",
    }[lang]
    prompt = intro
    for i, row in enumerate(examples.itertuples(), 1):
        prompt += fmt.format(i=i, a=row._1, h=row._2)
    prompt += q.format(act=activity)
    return prompt

def prompt_risk(examples: pd.DataFrame, activity: str, hazard: str, lang: str) -> str:
    json_tpl = {
        "Korean": "{\"빈도\": 숫자, \"강도\": 숫자, \"T\": 숫자}",
        "English": "{\"frequency\": number, \"intensity\": number, \"T\": number}",
        "Chinese": "{\"频率\": 数字, \"强度\": 数字, \"T\": 数字}",
    }[lang]
    fmt_ex = {
        "Korean": "예시 {i}: 입력: {inp}\n출력: {out}\n\n",
        "English": "Example {i}: Input: {inp}\nOutput: {out}\n\n",
        "Chinese": "示例 {i}: 输入: {inp}\n输出: {out}\n\n",
    }[lang]
    prompt = ""
    for i, row in enumerate(examples.itertuples(), 1):
        inp = f"{row._1} - {row._2}"
        out = f"{{\"빈도\": {row.빈도}, \"강도\": {row.강도}, \"T\": {row.T}}}"
        prompt += fmt_ex.format(i=i, inp=inp, out=out)
    q = {
        "Korean": "입력: {a} - {h}\n위 입력을 바탕으로 빈도·강도·T를 예측하고 다음 형식(JSON)으로 출력하세요:\n{tpl}\n출력:",
        "English": "Input: {a} - {h}\nPredict frequency, intensity, and T then output JSON:\n{tpl}\nOutput:",
        "Chinese": "输入: {a} - {h}\n预测频率、强度、T 并以 JSON 输出：\n{tpl}\n输出:",
    }[lang]
    prompt += q.format(a=activity, h=hazard, tpl=json_tpl)
    return prompt

def prompt_improvement(examples: pd.DataFrame, activity: str, hazard: str, f: int, i_: int, t: int, lang: str) -> str:
    # Only 2 examples to keep tokens low
    def _json(freq_b, int_b, plan):
        return f"{{\"개선대책\": \"{plan}\", \"개선 후 빈도\": 1, \"개선 후 강도\": 2, \"개선 후 T\": 2, \"T 감소율\": 80.0}}"

    examples_txt = ""
    for k, row in examples.head(2).iterrows():
        examples_txt += (
            f"Example:\nInput: {row['작업활동 및 내용']} / {row['유해위험요인 및 환경측면 영향']} / F={row['빈도']} / I={row['강도']} / T={row['T']}\n"
            f"Output(JSON): {_json(row['빈도'], row['강도'], '작업 구역 3m 앞 펜스 설치 등')}\n\n"
        )
    body = (
        f"Now provide a **specific, engineering‑level improvement plan** for the new input and quantify risk reduction.\n"
        f"Input: {activity} / {hazard} / F={f} / I={i_} / T={t}\n"
        f"Return **only valid JSON** with keys: 개선대책, 개선 후 빈도, 개선 후 강도, 개선 후 T, T 감소율."
    )
    return examples_txt + body

# -----------------------------------------------------------------------------
# 9. MAIN RUN BLOCK
# -----------------------------------------------------------------------------
if run and user_activity and api_key:

    # 9‑1. Load & embed dataset (cached on api_key)
    dataset_df = load_dataset()
    if st.session_state.faiss_index is None:
        with st.spinner("Preparing embeddings ..."):
            idx, emb_arr = build_embeddings(dataset_df, api_key)
            st.session_state.faiss_index = idx
            st.session_state.retriever_df = dataset_df
            st.session_state.embeddings = emb_arr
    else:
        idx = st.session_state.faiss_index
        dataset_df = st.session_state.retriever_df

    # 9‑2. Retrieve top‑3 similar rows
    openai.api_key = api_key
    q_emb = openai.Embedding.create(model="text-embedding-3-large", input=[user_activity])["data"][0]["embedding"]
    D, I = idx.search(np.array([q_emb], dtype="float32"), 3)
    retrieved = dataset_df.iloc[I[0]]

    # 9‑3. HAZARD prediction
    haz_prompt = prompt_hazard(retrieved[["작업활동 및 내용", "유해위험요인 및 환경측면 영향"]], user_activity, st.session_state.lang)
    hazard = gpt_chat(haz_prompt, api_key, st.session_state.lang, 120)

    # 9‑4. RISK numbers
    risk_prompt = prompt_risk(retrieved[["작업활동 및 내용", "유해위험요인 및 환경측면 영향", "빈도", "강도", "T"]],
                              user_activity, hazard, st.session_state.lang)
    risk_json = gpt_chat(risk_prompt, api_key, st.session_state.lang, 120)
    match = re.search(r"([1-5]).*?([1-5]).*?(\d+)", risk_json)
    freq = int(match.group(1)) if match else 3
    inten = int(match.group(2)) if match else 3
    t_val = int(match.group(3)) if match else freq * inten
    grade = determine_grade(t_val, st.session_state.lang)

    # 9‑5. IMPROVEMENT plan
    imp_prompt = prompt_improvement(retrieved, user_activity, hazard, freq, inten, t_val, st.session_state.lang)
    imp_json_raw = gpt_chat(imp_prompt, api_key, st.session_state.lang, 200)
    try:
        imp_data = json.loads(re.sub("```[a-z]*", "", imp_json_raw))
    except Exception:
        imp_data = {}
    imp_plan = imp_data.get("개선대책", imp_json_raw)
    imp_freq = imp_data.get("개선 후 빈도", 1)
    imp_inten = imp_data.get("개선 후 강도", 2)
    imp_t = imp_data.get("개선 후 T", imp_freq * imp_inten)
    imp_rrr = imp_data.get("T 감소율", round((t_val - imp_t) * 100 / t_val, 2))

    # 9‑6. DISPLAY RESULTS ------------------------------------------------------
    st.markdown("## " + TXT["result_header"])

    # risk table fixed width, scrollable rows
    st.write("### AI Risk Assessment")
    assess_df = pd.DataFrame({
        "작업활동": [user_activity],
        "유해위험요인": [hazard],
        "빈도": [freq],
        "강도": [inten],
        "T": [t_val],
        TXT["grade"]: [grade],
    })
    st.dataframe(assess_df, use_container_width=True)

    # improvement
    st.write("### " + TXT["improvement_header"])
    imp_df = pd.DataFrame({
        "항목": ["빈도", "강도", "T"],
        "개선 전": [freq, inten, t_val],
        "개선 후": [imp_freq, imp_inten, imp_t],
    })
    st.dataframe(imp_df, use_container_width=True)
    st.success(f"**RRR:** {imp_rrr}%")
    st.markdown(f"**개선대책:**\n{imp_plan}")

    # similar cases
    st.write("### " + TXT["similar_cases"])
    st.dataframe(retrieved[["작업활동 및 내용", "유해위험요인 및 환경측면 영향", "T", "등급"]], height=180)

    st.session_state.last_result = {
        "assessment": assess_df,
        "improvement": imp_df,
    }
else:
    st.info("⬅️  사이드바에 API Key와 작업활동을 입력 후 **Run** 을 누르세요.")
