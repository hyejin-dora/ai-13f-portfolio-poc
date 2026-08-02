"""AI 13F 포트폴리오 분석 PoC - 메인 화면.

이번 단계에서는 프로젝트 소개 화면만 표시합니다.
SEC / Gemini API 호출 기능은 다음 단계에서 추가합니다.
"""

import streamlit as st

st.set_page_config(page_title="AI 13F 포트폴리오 분석 PoC", page_icon="📊")

st.title("📊 AI 13F 포트폴리오 분석 PoC")

st.subheader("분석 대상")
st.write("Berkshire Hathaway (CIK 0001067983)")

st.subheader("프로젝트 설명")
st.write(
    "미국 증권거래위원회(SEC)에 제출되는 13F 공시 데이터를 불러와 "
    "기관투자자의 주식 보유 현황과 분기별 변화를 분석하고, "
    "그 결과를 Gemini API가 자연어로 쉽게 설명해 주는 웹 애플리케이션입니다."
)

st.info(
    "본 프로젝트는 교육용 PoC(개념 검증)입니다. "
    "투자 자문이나 투자 권유가 아니며, 분석 결과의 정확성을 보장하지 않습니다."
)
