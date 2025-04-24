import streamlit as st
import openai
import numpy as np
import json
import os

# Set page configuration for wide layout and title
st.set_page_config(page_title="Artificial Intelligence Risk Assessment", layout="wide")


st.sidebar.markdown("## 🔑 OpenAI API Key")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# OpenAI 호출 전 API 키 검증
if not user_api_key:
    st.warning("⚠️ Please input your OpenAI API Key at left.")
    st.stop()

# Mapping for language selection (display text to code)
lang_map = {
    "🇰🇷 Korean": "ko",
    "🇺🇸 English": "en",
    "🇨🇳 Chinese": "zh"
}

# Prepare text strings in different languages for UI elements
app_title = {
    "ko": "인공지능 위험성 평가", 
    "en": "Artificial Intelligence Risk Assessment", 
    "zh": "人工智能风险评估"
}
input_labels = {
    "category": {"ko": "공종", "en": "Construction Category", "zh": "施工类别"},
    "activity": {"ko": "작업활동 및 내용", "en": "Work Activity & Details", "zh": "工作活动及内容"},
    "submit": {"ko": "위험성 평가 실행", "en": "Run Risk Assessment", "zh": "执行风险评估"},
    "similar_cases": {"ko": "유사 사례", "en": "Similar Cases", "zh": "类似案例"}
}
category_options = {
    "ko": ["건축", "토목", "플랜트"],
    "en": ["Building", "Civil", "Plant"],
    "zh": ["建筑", "土木", "工厂"]
}
placeholder_text = {
    "ko": "예: 5미터 높이 비계 작업 중 자재 인양",
    "en": "e.g., Lifting materials on a 5m high scaffold",
    "zh": "例如: 在5米高的脚手架上提升材料"
}
results_title = {
    "ko": "AI 위험성 평가 결과",
    "en": "AI Risk Assessment",
    "zh": "AI风险评估结果"
}
summary_template = {
    "ko": (
        "이 작업의 위험성을 평가한 결과, **{hazard}**으로 인한 사고 발생 위험이 "
        "**{grade} 등급**으로 나타났습니다. 또한, 유사한 중대재해 사례가 총 **{count}건** 확인되었습니다."
    ),
    "en": (
        "The risk assessment indicates that the risk of an accident due to **{hazard}** "
        "is at a **{grade}** level. Additionally, a total of **{count}** similar serious accident cases were found."
    ),
    "zh": (
        "风险评估显示，由于**{hazard}**导致事故发生的风险等级为**{grade}级**。"
        "另外，共发现**{count}**起类似的重大事故案例。"
    )
}

# Set up OpenAI API key (from secrets or environment)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    st.warning("OpenAI API key is not set. Please add it to Streamlit secrets or environment variable.")

# Layout: Title on left, language selector on right
col1, col2 = st.columns([9, 1])
with col1:
    # Display the app title in the selected language (synced with language choice)
    selected_lang = st.session_state.get("selected_lang", "🇰🇷 Korean")
    st.title(app_title[lang_map[selected_lang]])
with col2:
    # Language selector at top-right with flag icons
    selected_lang = st.selectbox("Language", list(lang_map.keys()), index=list(lang_map.keys()).index(selected_lang))
    # Remember choice in session state for persistence
    st.session_state["selected_lang"] = selected_lang

# Determine language code from selection (e.g., "ko", "en", "zh")
lang_code = lang_map[selected_lang]

# Columns for input (left) and output (right)
col_input, col_output = st.columns([1, 3])
with col_input:
    # Input Form for category and work activity
    with st.form("input_form"):
        # Category selection (공종)
        st.markdown(f"**{input_labels['category'][lang_code]}**")  # Show the label in bold
        category = st.selectbox(
            "", 
            options=category_options[lang_code], 
            index=0, 
            key="category_select"
        )
        # Work activity description text area
        description = st.text_area(
            input_labels["activity"][lang_code], 
            value="", 
            placeholder=placeholder_text[lang_code], 
            key="activity_input"
        )
        # Submit button triggers the risk assessment
        submitted = st.form_submit_button(input_labels["submit"][lang_code])

# When the form is submitted, perform the AI analysis (Phase 1 & 2)
if submitted:
    # Combine category and description for AI prompt context
    task_description = (
        f"{input_labels['category'][lang_code]}: {category}\n"
        f"{input_labels['activity'][lang_code]}: {description}"
    )
    # **Phase 1** – Hazard Prediction and Risk Assessment
    try:
        if lang_code == "ko":
            # Korean prompt for hazard identification
            system_msg = (
                "당신은 건설 안전 분야의 전문가 AI입니다. 주어진 작업에 대해 발생 가능한 유해위험요인을 식별하고, "
                "각 요인의 위험 수준을 A부터 E까지 등급으로 평가하세요. (A: 가장 높은 위험, E: 가장 낮은 위험) "
                "개선 대책은 이 단계에서 제공하지 마세요."
            )
            user_msg = (
                task_description + 
                "\n위의 작업에 대한 유해위험요인과 개선 전 위험등급(A~E)을 JSON 형식으로만 보여주세요. "
                "키는 'hazard'와 'risk_grade'로 해주세요."
            )
        elif lang_code == "zh":
            # Chinese prompt for hazard identification
            system_msg = (
                "你是一位建筑施工安全专家AI。请识别给定工作中的潜在危险因素，并评估每个危险因素在"
                "没有任何改进措施时的风险等级（A至E，A为最高风险，E为最低风险）。在此阶段不需要提供改进措施。"
            )
            user_msg = (
                task_description + 
                "\n请以JSON格式返回这些危险因素及其风险等级。JSON数组中的每个元素应包含 'hazard' 和 'risk_grade' 字段。"
                "只输出JSON，不要添加额外说明。"
            )
        else:
            # English prompt for hazard identification
            system_msg = (
                "You are a construction safety expert AI. Identify potential hazards for the given task and assess the "
                "risk level of each hazard before any improvements. Risk levels should be graded from A (highest risk) to "
                "E (lowest risk). Do not provide any improvement measures yet."
            )
            user_msg = (
                task_description + 
                "\nList the hazards and their risk grade (A-E) in JSON format as an array of objects with keys 'hazard' and 'risk_grade'. "
                "Provide only the JSON output with no extra text."
            )
        # Call OpenAI API for hazard identification
        response1 = openai.ChatCompletion.create(
            model="gpt-4o",  # or "gpt-4" if available
            api_key=user_api_key,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        hazards_content = response1.choices[0].message.content.strip()
        # Remove any Markdown formatting (like ```json``` wrappers) from the response
        if hazards_content.startswith("```"):
            hazards_content = hazards_content.strip("```").strip()
            if hazards_content.lower().startswith("json"):
                hazards_content = hazards_content[len("json"):].strip()
        # Parse the JSON string to Python data
        hazards_data = json.loads(hazards_content)
    except Exception as e:
        st.error("Failed to retrieve hazard information. Please try again.")
        st.stop()
    # Validate that we got a list of hazards
    if not isinstance(hazards_data, list):
        st.error("Unexpected hazards format from AI. Please try again.")
        st.stop()

    # **Phase 2** – Generate Improvement Measures for each identified hazard
    try:
        if lang_code == "ko":
            # Prepare hazard list text for Korean prompt
            hazard_list_text = "\n".join([
                f"- {item['hazard']} (등급: {item['risk_grade']})" 
                for item in hazards_data
            ])
            system_msg2 = (
                "당신은 건설 안전 전문가 AI입니다. 이제 각 위험요인에 대한 개선 대책을 구체적으로 제시하고 "
                "개선 후 위험등급을 평가하세요."
            )
            user_msg2 = (
                f"식별된 위험요인 목록:\n{hazard_list_text}\n"
                "각 위험요인에 대해 실행 가능한 구체적인 개선대책을 제안하고, 개선 조치 시행 후의 새로운 위험등급(A~E)을 알려주세요. "
                "결과를 JSON 형식으로만 제공해주세요. 각 요소는 'hazard', 'risk_grade_before', 'improvement', 'risk_grade_after' 키를 포함해야 합니다."
            )
        elif lang_code == "zh":
            # Prepare hazard list text for Chinese prompt
            hazard_list_text = "\n".join([
                f"- {item['hazard']} (等级: {item['risk_grade']})" 
                for item in hazards_data
            ])
            system_msg2 = "你是一位建筑安全专家AI。请为每个危险因素提供改进措施。"
            user_msg2 = (
                f"已识别的危险因素列表:\n{hazard_list_text}\n"
                "请针对上述每个危险因素提供具体可行的改进措施，并给出实施改进后的新的风险等级（A至E）。"
                "请以JSON格式返回结果，每个元素包含 'hazard', 'risk_grade_before', 'improvement', 'risk_grade_after' 字段。"
            )
        else:
            # Prepare hazard list text for English prompt
            hazard_list_text = "\n".join([
                f"- {item['hazard']} (Grade: {item['risk_grade']})" 
                for item in hazards_data
            ])
            system_msg2 = (
                "You are a construction safety expert AI. Provide improvement measures for each identified hazard."
            )
            user_msg2 = (
                f"Identified hazards:\n{hazard_list_text}\n"
                "For each hazard listed above, suggest a specific and actionable improvement measure to mitigate it, "
                "and provide the new risk grade (A-E) after implementing the improvement. "
                "Respond in JSON format as an array of objects, each with 'hazard', 'risk_grade_before', 'improvement', 'risk_grade_after'."
            )
        # Call OpenAI API for improvement measures
        response2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for potentially better output
            messages=[
                {"role": "system", "content": system_msg2},
                {"role": "user", "content": user_msg2}
            ]
        )
        improvement_content = response2.choices[0].message.content.strip()
        # Strip any Markdown formatting from the response
        if improvement_content.startswith("```"):
            improvement_content = improvement_content.strip("```").strip()
            if improvement_content.lower().startswith("json"):
                improvement_content = improvement_content[len("json"):].strip()
        improvements_data = json.loads(improvement_content)
    except Exception as e:
        st.error("Failed to retrieve improvement measures. Please try again.")
        st.stop()
    # Validate that we got a list of improvements
    if not isinstance(improvements_data, list):
        st.error("Unexpected improvements format from AI. Please try again.")
        st.stop()

    # **Summary & Results Display**

    # Determine the highest-risk hazard identified (for summary sentence)
    risk_rank = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}  # higher number = higher risk
    highest_hazard = max(hazards_data, key=lambda x: risk_rank.get(x.get("risk_grade", "E"), 1)) if hazards_data else None
    hazard_name = highest_hazard.get("hazard", "") if highest_hazard else ""
    risk_grade = highest_hazard.get("risk_grade", "") if highest_hazard else ""

    # Retrieve similar past cases using cosine similarity on embeddings (retrieved_docs)
    # Note: In a real app, pre-compute and store embeddings for a large case database for efficiency.
    example_cases = {
        "ko": [
            "건설 현장에서 작업자가 비계에서 추락하여 부상",
            "굴착 작업 중 토사가 붕괴됨",
            "크레인 작업에서 물체가 낙하함"
        ],
        "en": [
            "Worker fell from scaffolding at a construction site",
            "Soil collapsed during an excavation work",
            "An object fell during a crane operation"
        ],
        "zh": [
            "建筑工地的工人从脚手架上坠落受伤",
            "挖掘作业中土方坍塌",
            "起重机作业时物体坠落"
        ]
    }
    case_texts = example_cases.get(lang_code, [])
    similar_cases = []
    similar_count = 0
    if case_texts:
        try:
            # Create embeddings for the user input description and example cases
            embed_input = [description] + case_texts
            embed_response = openai.Embedding.create(model="text-embedding-ada-002", input=embed_input)
            embeddings = [item["embedding"] for item in embed_response["data"]]
            query_vec = embeddings[0]
            case_vecs = embeddings[1:]
            # Compute cosine similarity between input and each case
            query_norm = np.linalg.norm(query_vec)
            similarities = []
            for vec in case_vecs:
                sim_val = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec))
                similarities.append(sim_val)
            # Filter cases with similarity >= 0.70
            for sim_val, text in sorted(zip(similarities, case_texts), key=lambda x: x[0], reverse=True):
                if sim_val >= 0.70:
                    similar_cases.append((text, sim_val))
            similar_count = len(similar_cases)
            # (If no case meets the threshold, similar_cases will remain empty and count = 0)
        except Exception as e:
            # If embedding fails, proceed without retrieved cases
            similar_cases = []
            similar_count = 0

    # Display the results in the output column
    with col_output:
        # Section title (localized)
        st.subheader(results_title[lang_code])
        # Summary sentence with main hazard and risk grade
        summary_text = summary_template[lang_code].format(hazard=hazard_name, grade=risk_grade, count=similar_count)
        st.markdown(summary_text)
        # Table of hazards, improvements, and risk grades
        table_rows = []
        for item in improvements_data:
            hazard_factor = item.get("hazard", "")
            before_grade = item.get("risk_grade_before", "")
            improvement_measure = item.get("improvement", "")
            after_grade = item.get("risk_grade_after", "")
            table_rows.append({
                # Column headers are localized based on selected language
                "유해위험요인" if lang_code == "ko" else ("危险因素" if lang_code == "zh" else "Hazard Factor"): hazard_factor,
                "개선전 등급" if lang_code == "ko" else ("改进前等级" if lang_code == "zh" else "Risk Grade (Before)") : before_grade,
                "개선대책" if lang_code == "ko" else ("改进措施" if lang_code == "zh" else "Improvement Measure"): improvement_measure,
                "개선후 등급" if lang_code == "ko" else ("改进后等级" if lang_code == "zh" else "Risk Grade (After)") : after_grade
            })
        st.table(table_rows)
        # List of similar past cases (if any found above similarity threshold)
        if similar_cases:
            st.subheader(input_labels["similar_cases"][lang_code])
            for text, sim_val in similar_cases:
                # Display each case with its cosine similarity score
                st.write(f"- ({sim_val:.2f}) {text}")
# If form not submitted yet, show an instructional message in the output area
else:
    with col_output:
        st.info("➤ Please enter the work activity details on the left, then click **Run Risk Assessment** to see results.")
