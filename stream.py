"""
Streamlit App: Integrated AI Risk Assessment (Phase 1 + Phase 2)
----------------------------------------------------------------
* Single input → full pipeline (hazard prediction ➝ risk grading ➝ improvement measures)
* Multilingual UI (Korean / English / Chinese)
* No artificial embedding‑count limits or demo warning messages
* Phase 2 prompt examples explicitly include the “Improvement Plan” field
"""
# ---------- Imports ----------
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split
# ---------- Language Pack ----------
# (trimmed to the fields referenced in the new UI; you can freely extend)
LANG = {
    "Korean": {
        "title": "AI 위험성평가 통합 시스템",
        "api_key": "OpenAI API 키 입력",
        "dataset": "데이터셋 선택",
        "load_btn": "데이터 로드 및 인덱스 구축",
        "loading": "데이터 로드 및 인덱스 구축 중…",
        "loaded": "데이터 로드 및 인덱스 구축 완료! (총 {n}개)",
        "work_input": "작업활동 입력",
        "run_btn": "위험성 평가 및 개선대책 생성",
        "similar_cases": "유사 사례",
        "prediction": "예측 결과",
        "hazard": "예측된 유해위험요인",
        "risk_table": ["항목", "값"],
        "risk_rows": ["빈도", "강도", "T 값", "등급"],
        "improvement_header": "개선대책",
        "risk_change": "위험도(T) 변화",
        "before": "개선 전 T값",
        "after": "개선 후 T값",
        "rrr": "위험 감소율 (RRR)",
        "api_warn": "API 키를 입력하세요.",
        "load_warn": "먼저 데이터셋을 로드하세요.",
        "input_warn": "작업활동을 입력하세요.",
    },
    "English": {
        "title": "AI Risk‑Assessment Integrated System",
        "api_key": "Enter OpenAI API Key",
        "dataset": "Select Dataset",
        "load_btn": "Load Data & Build Index",
        "loading": "Loading data & building index…",
        "loaded": "Data loaded & index built! (total {n})",
        "work_input": "Work Activity",
        "run_btn": "Run Risk‑Assessment & Improvement",
        "similar_cases": "Similar Cases",
        "prediction": "Prediction Result",
        "hazard": "Predicted Hazard",
        "risk_table": ["Item", "Value"],
        "risk_rows": ["Frequency", "Intensity", "T Value", "Grade"],
        "improvement_header": "Improvement Plan",
        "risk_change": "T‑value Change",
        "before": "T‑value Before",
        "after": "T‑value After",
        "rrr": "Risk Reduction Rate (RRR)",
        "api_warn": "Please enter an API key.",
        "load_warn": "Load a dataset first.",
        "input_warn": "Please enter a work activity.",
    },
    "Chinese": {
        "title": "AI风险评估一体化系统",
        "api_key": "输入 OpenAI API 密钥",
        "dataset": "选择数据集",
        "load_btn": "加载数据并建立索引",
        "loading": "数据加载与索引构建中…",
        "loaded": "数据加载与索引构建完成！（共 {n} 条）",
        "work_input": "工作活动",
        "run_btn": "执行风险评估与改进",
        "similar_cases": "相似案例",
        "prediction": "预测结果",
        "hazard": "预测危害",
        "risk_table": ["项目", "值"],
        "risk_rows": ["频率", "强度", "T 值", "等级"],
        "improvement_header": "改进措施",
        "risk_change": "T 值变化",
        "before": "改进前 T 值",
        "after": "改进后 T 值",
        "rrr": "风险降低率 (RRR)",
        "api_warn": "请输入 API 密钥。",
        "load_warn": "请先加载数据集。",
        "input_warn": "请输入工作活动。",
    },
}
# ---------- Page Config ----------
st.set_page_config("AI Risk Assessment", ":망치와_렌치:", layout="wide")
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
for key, default in {
    "lang": "Korean", "index": None, "df": None, "api_key": ""}.items():
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
                ss.df = _load_data(dataset_name)  # function defined later
                ss.index = _build_index(ss.df, ss.api_key)  # function defined later
            st.success(txt["loaded"].format(n=len(ss.df)))
# ---------- Main Layout ----------
st.markdown(f"<p class='title'>{txt['title']}</p>", unsafe_allow_html=True)
work_activity = st.text_input(txt["work_input"], key="work")
run = st.button(txt["run_btn"], key="run")
# ---------- Helper: Grade ----------
GRADE = [(16,25,'A'),(10,15,'B'),(5,9,'C'),(3,4,'D'),(1,2,'E')]
def grade(t):
    for lo,hi,g in GRADE:
        if lo<=t<=hi: return g
    return "?"
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
        # 1) Retrieve similar examples
        query_emb = _embed([work_activity], ss.api_key)[0]  # helper
        D,I = ss.index.search(np.array([query_emb], dtype='float32'), min(3,len(ss.df)))
        sims = ss.df.iloc[I[0]]
        # show similar cases
        st.subheader(txt["similar_cases"])
        for i,row in sims.iterrows():
            st.markdown(
                f"<div class='similar-case'><b>#{i}</b><br>작업활동: {row['작업활동 및 내용']}<br>유해위험요인: {row['유해위험요인 및 환경측면 영향']}<br>위험도: {row['T']} (빈도 {row['빈도']}, 강도 {row['강도']}, 등급 {row['등급']})</div>",
                unsafe_allow_html=True)
        # 2) Phase‑1 prompts
        hz_prompt = _prompt_hazard(sims, work_activity)  # helper
        hazard = _ask_gpt(hz_prompt, ss.api_key)
        risk_prompt = _prompt_risk(sims, work_activity, hazard)  # helper
        risk_raw = _ask_gpt(risk_prompt, ss.api_key)
        freq,intensity,T = _parse_risk(risk_raw)
        # 3) Display Phase‑1 outputs
        st.subheader(txt["prediction"])
        st.write(f"**{txt['hazard']}**: {hazard}")
        st.table(pd.DataFrame({txt['risk_table'][0]:txt['risk_rows'],
                              txt['risk_table'][1]:[freq,intensity,T,grade(T)]}))
        # 4) Phase‑2 prompt & answer
        imp_prompt = _prompt_improve(sims, work_activity, hazard, freq, intensity, T)  # helper
        imp_raw = _ask_gpt(imp_prompt, ss.api_key)
        imp_parsed = _parse_improve(imp_raw)
        imp_plan = imp_parsed.get("plan","-")
        newF,newI,newT = imp_parsed.get("f",1),imp_parsed.get("i",1),imp_parsed.get("t",1)
        rrr = imp_parsed.get("rrr", (T-newT)/T*100 if T else 0)
        # 5) Display Phase‑2 outputs
        st.subheader(txt['improvement_header'])
        st.markdown(f"<div class='box'>{imp_plan}</div>", unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        col1.metric(txt['before'], T)
        col2.metric(txt['after'], newT, delta=f"{rrr:.1f}%")
# ---------- Utility Functions ----------
def _load_data(name:str)->pd.DataFrame:
    """Load xlsx ↦ tidy DataFrame, compute T & grade."""
    try:
        df = pd.read_excel(f"{name}.xlsx")
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]:'빈도', df.columns[5]:'강도', df.columns[6]:'T'})
        df['T'] = pd.to_numeric(df['빈도'])*pd.to_numeric(df['강도'])
        df['등급'] = df['T'].apply(grade)
        df['content'] = df.apply(lambda r:' '.join(r.astype(str)), axis=1)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        # fallback dummy
        data={"작업활동 및 내용":["Excavation"],"유해위험요인 및 환경측면 영향":["Collapse"],"빈도":[3],"강도":[4]}
        df=pd.DataFrame(data)
        df['T']=df['빈도']*df['강도'];df['등급']=df['T'].apply(grade);df['content']=df.apply(lambda r:' '.join(r.astype(str)),axis=1)
        return df
def _embed(texts,list_api_key):
    openai.api_key=list_api_key
    embs=[]
    for t in texts:
        r=openai.Embedding.create(model="text-embedding-3-large",input=[t.replace("\n"," ")])
        embs.append(r['data'][0]['embedding'])
    return embs
def _build_index(df, api_key):
    embs=_embed(df['content'].tolist(), api_key)
    arr=np.array(embs, dtype='float32')
    index=faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index
# ----- Prompt builders & parsers (simplified) -----
def _prompt_hazard(docs, activity):
    examples="".join([
        f"예시 {i+1}:\n작업활동: {r['작업활동 및 내용']}\n유해위험요인: {r['유해위험요인 및 환경측면 영향']}\n\n"
        for i,(_,r) in enumerate(docs.iterrows()) ])
    return f"""다음은 건설현장 작업활동과 유해위험요인 예시입니다.\n\n{examples}다음 작업활동의 유해위험요인을 예측하세요:\n작업활동: {activity}\n유해위험요인:"""
def _prompt_risk(docs, activity, hazard):
    examples="".join([
        f"예시 {i+1}:\n입력: {r['작업활동 및 내용']} - {r['유해위험요인 및 환경측면 영향']}\n출력: {{\"빈도\": {r['빈도']}, \"강도\": {r['강도']}, \"T\": {r['T']}}}\n\n"
        for i,(_,r) in enumerate(docs.iterrows()) ])
    return f"""{examples}입력: {activity} - {hazard}\n빈도와 강도를 1~5 정수로 예측하고 JSON으로 출력: {{\"빈도\": n, \"강도\": n, \"T\": n}}\n출력:"""
def _prompt_improve(docs, activity, hazard, f, i, t):
    # Include improvement plan in examples
    examples="""
Example:
Input (Activity): Excavation
Input (Hazard): Wall collapse
Input (Original F/I/T): 3/4/12
Output (Improvement) JSON:
{"""+"\n  \"개선대책\": \"1) 토양 경사 준수 2) 지보공 설치 3) 점검\",\n  \"개선 후 빈도\": 1,\n  \"개선 후 강도\": 2,\n  \"개선 후 T\": 2,\n  \"T 감소율\": 83.33\n}"""
    return f"""{examples}\n새로운 입력:\nInput (Activity): {activity}\nInput (Hazard): {hazard}\nInput (Original F/I/T): {f}/{i}/{t}\nJSON key: 개선대책, 개선 후 빈도, 개선 후 강도, 개선 후 T, T 감소율\n번호 매긴 개선대책을 한국어로 3개 이상 포함하고 올바른 JSON만 출력하세요.\n출력:"""
def _ask_gpt(prompt, api_key, model="gpt-4o"):
    openai.api_key=api_key
    res=openai.ChatCompletion.create(model=model,temperature=0, messages=[{"role":"user","content":prompt}])
    return res['choices'][0]['message']['content']
def _parse_risk(txt):
    m=re.search(r'\{"빈도"\s*:\s*(\d),\s*"강도"\s*:\s*(\d),\s*"T"\s*:\s*(\d+)\}',txt)
    return (int(m.group(1)),int(m.group(2)),int(m.group(3))) if m else (3,3,9)
def _parse_improve(txt):
    try:
        j=re.search(r'\{.*\}',txt,re.S).group()
        data=pd.json.loads(j)
    except:
        data={}
    return {
        "plan": data.get("개선대책",""),
        "f": data.get("개선 후 빈도",1),
        "i": data.get("개선 후 강도",1),
        "t": data.get("개선 후 T",1),
        "rrr": data.get("T 감소율",0),
    }
# ---------- Footer ----------
footer_cols = st.columns([1,1])
for path,col in zip(["cau.png","doosan.png"],footer_cols):
    if os.path.exists(path):
        col.image(Image.open(path), width=140)






