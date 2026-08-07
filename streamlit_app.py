"""AI 13F 포트폴리오 분석 PoC - 메인 화면.

SEC EDGAR에서 최근 13F-HR 공시를 조회해 보유 종목과 두 분기 변화를 보여주고,
그 분석 결과를 Gemini API가 한국어 리서치 브리핑으로 설명해 줍니다.

화면(이 파일)은 사용자에게 보여주는 일만 하고,
데이터 수집은 services/sec_client.py, 계산은 services/portfolio_analysis.py,
브리핑 생성은 services/llm_client.py가 담당합니다.

SEC 조회 결과를 잠시 보관해 두고 재사용하는 캐시도 이 파일이 맡습니다.
캐시는 Streamlit 전용 기능(st.cache_data)이므로, 수집 로직이 들어 있는
services/sec_client.py는 캐시를 모른 채 순수한 조회 함수로 남습니다.
"""

import base64
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from services.institution_comparison import (
    HOLDING_TYPE,
    HOLDING_TYPE_COMMON,
    HOLDING_TYPE_LEFT_ONLY,
    HOLDING_TYPE_RIGHT_ONLY,
    build_institution_comparison_briefing_payload,
    compare_institution_portfolios,
    find_common_report_dates,
    index_filings_by_report_date,
    summarize_institution_comparison,
)
from services.llm_client import (
    LlmApiError,
    build_briefing_prompt,
    build_institution_comparison_briefing_prompt,
    generate_briefing,
)
from services.portfolio_analysis import (
    STATUS_DECREASED,
    STATUS_EXITED,
    STATUS_INCREASED,
    STATUS_NEW,
    STATUS_UNCHANGED,
    compare_holdings,
    summarize_comparison,
)
from services.sec_client import (
    SecApiError,
    get_13f_holdings,
    get_recent_13f_filings,
)

# 화면을 처음 열었을 때 기본으로 선택되는 운용사.
# 목록 자체는 data/managers.csv에서 읽어오고, 사용자가 다른 기관으로 바꿀 수 있습니다.
DEFAULT_MANAGER = "Berkshire Hathaway"

# 기관을 바꿀 때 지워야 하는 화면 상태(세션 상태) 키 목록.
# 이전 기관의 공시·보유 종목·분기 비교·AI 브리핑 결과가 화면에 남지 않게 하기 위함입니다.
ANALYSIS_STATE_KEYS = (
    "filings",
    "error",
    "holdings",
    "holdings_error",
    "holdings_filing",
    "comparison",
    "comparison_summary",
    "comparison_filings",
    "comparison_error",
    "comparison_warning",
    "ai_briefing",
    "ai_briefing_error",
)

# 조회할 공시 건수.
FILING_LIMIT = 2

# 기관 간 비교 화면에서 기관별로 조회할 공시 건수.
# 기관마다 13F 제출을 시작한 시점과 제출 주기가 달라, 최근 2건만 보면 두 기관이
# 함께 공시한 분기(report_date)를 못 찾을 수 있습니다. 그래서 이 화면에서만
# 조회 범위를 넓혀 최근 8건 안에서 공통 분기를 찾습니다.
# (위 FILING_LIMIT과 단일 기관 분석 동작은 그대로 둡니다.)
INSTITUTION_COMPARISON_FILING_LIMIT = 8

# 기관 간 비교 화면에서 기본으로 선택되는 두 기관.
# 목록에 없으면 아래 institution_default_index가 다른 기관으로 대체합니다.
DEFAULT_LEFT_INSTITUTION = DEFAULT_MANAGER
DEFAULT_RIGHT_INSTITUTION = "Pershing Square Capital Management"

# 기관 비교 AI 브리핑 결과만 담는 화면 상태 키.
# 단일 기관 브리핑(ai_briefing / ai_briefing_error)과 이름을 나누어 두었기 때문에,
# 한쪽 브리핑을 지워도 다른 쪽은 화면에 그대로 남습니다.
INSTITUTION_AI_BRIEFING_STATE_KEYS = (
    "institution_ai_briefing",
    "institution_ai_briefing_error",
)

# 기관 간 비교 화면에서만 쓰는 화면 상태 키.
# 위 ANALYSIS_STATE_KEYS(단일 기관 분석)와 겹치지 않게 이름을 나누어 두었기 때문에,
# 비교할 기관을 바꿔도 단일 기관 분석 결과는 지워지지 않습니다.
INSTITUTION_COMPARISON_STATE_KEYS = (
    "institution_common_dates",
    "institution_left_filings_by_date",
    "institution_right_filings_by_date",
    "institution_selected_report_date",
    "institution_comparison",
    "institution_comparison_summary",
    "institution_comparison_filings",
    "institution_comparison_error",
    "institution_comparison_warning",
    "institution_comparison_left_name",
    "institution_comparison_right_name",
    # 비교 결과가 바뀌면 그 결과로 만든 AI 브리핑도 함께 지워야 하므로 포함합니다.
    *INSTITUTION_AI_BRIEFING_STATE_KEYS,
)

# 비중 상위 몇 개 종목을 따로 보여줄지.
TOP_HOLDINGS_COUNT = 10

# SEC 조회 결과를 얼마나 오래 재사용할지(초). 6시간.
# 13F는 분기마다 나오는 공시라 하루 안에 내용이 바뀌는 일이 거의 없으므로,
# 같은 값을 반복해서 내려받지 않도록 넉넉하게 잡았습니다.
SEC_CACHE_TTL_SECONDS = 6 * 60 * 60

# 행이 많은 표(예: 보유 종목이 수백 개인 기관)의 표시 높이(픽셀).
# 높이를 정해 두면 표 안에서만 스크롤되어 화면이 지나치게 길어지지 않습니다.
LARGE_TABLE_HEIGHT = 520

# 몇 행을 넘을 때부터 위 높이를 적용할지. 이보다 적으면 내용에 맞춘 높이를 써서
# 표 아래에 빈 공간이 생기지 않습니다.
LARGE_TABLE_ROW_THRESHOLD = 15

# 내용에 맞춰 높이를 정하라는 Streamlit 설정값.
# Streamlit은 표 높이로 양의 정수, "content", "stretch"만 허용합니다.
# (예전처럼 None을 넘기면 StreamlitInvalidHeightError가 납니다.)
AUTO_TABLE_HEIGHT = "content"

# 캐시를 비운 뒤 사용자에게 보여 줄 안내 문구.
CACHE_CLEARED_MESSAGE = (
    "SEC 조회 캐시를 비웠습니다. 최신 데이터를 받으려면 "
    "'최근 13F 공시 조회'와 '두 분기 변화 분석' 버튼을 다시 눌러 주세요."
)

MANAGERS_CSV = Path(__file__).parent / "data" / "managers.csv"

# 화면에 표시할 표의 열 순서와 한국어 제목.
COLUMN_LABELS = {
    "accession_number": "접수번호 (accession_number)",
    "filing_date": "제출일 (filing_date)",
    "report_date": "기준일 (report_date)",
    "primary_document": "주요 문서 (primary_document)",
}

# 보유 종목 표의 열 순서. sec_client.get_13f_holdings가 돌려주는 키를 그대로 씁니다.
HOLDINGS_COLUMNS = [
    "issuer_name",
    "class_title",
    "cusip",
    "reported_value",
    "shares",
    "share_type",
    "put_call",
]

# 숫자로 다뤄야 하는 열(합계와 비중 계산에 사용).
NUMERIC_HOLDINGS_COLUMNS = ["reported_value", "shares"]

# 글자로 다뤄야 하는 열(값이 없으면 빈칸으로 표시).
TEXT_HOLDINGS_COLUMNS = [
    "issuer_name",
    "class_title",
    "cusip",
    "share_type",
    "put_call",
]

# 분기 비교 표에 보여 줄 열 순서. 사람이 읽기 쉽게 종목명을 앞에 둡니다.
# (분석 결과 표의 열 이름은 그대로 쓰고, 순서만 바꿉니다.)
# put_call과 share_type은 같은 회사라도 보통주 보유와 옵션 보유가 별개의 줄로
# 나뉘는 이유를 알 수 있게 함께 보여 줍니다. 내부 식별용 position_key는
# 표에 넣지 않습니다.
COMPARISON_DISPLAY_COLUMNS = [
    "issuer_name",
    "class_title",
    "cusip",
    "put_call",
    "share_type",
    "previous_reported_value",
    "current_reported_value",
    "reported_value_change",
    "previous_shares",
    "current_shares",
    "shares_change",
    "previous_weight",
    "current_weight",
    "weight_change_pct_point",
    "change_status",
]

# 기관 간 비교 표에 보여 줄 열 순서.
# 금액은 기관마다 규모가 달라 그대로 비교하기 어려우므로, 비중(%)과 비중 차이(%포인트)를
# 함께 두어 규모와 관계없이 견줄 수 있게 합니다. 내부 식별용 position_key는 넣지 않습니다.
INSTITUTION_DISPLAY_COLUMNS = [
    "issuer_name",
    "cusip",
    "put_call",
    "share_type",
    "left_reported_value",
    "right_reported_value",
    "left_weight_pct",
    "right_weight_pct",
    "weight_gap_pct_point",
]

# 기관 간 비교 화면에서 탭으로 나눌 보유 유형 순서.
INSTITUTION_HOLDING_TYPE_ORDER = [
    HOLDING_TYPE_COMMON,
    HOLDING_TYPE_LEFT_ONLY,
    HOLDING_TYPE_RIGHT_ONLY,
]

# 기관 비교 AI 브리핑을 만드는 동안 보여 줄 문구.
INSTITUTION_BRIEFING_SPINNER_TEXT = (
    "기관 비교 결과를 바탕으로 AI 브리핑을 생성하고 있습니다."
)

# 기관 비교 AI 브리핑 영역에 함께 표시하는 안내 문구.
INSTITUTION_BRIEFING_NOTICE = (
    "AI 브리핑은 Python으로 계산된 기관 비교 결과를 설명하며, "
    "투자 추천이나 기관의 투자 의도 추정을 제공하지 않습니다."
)

# 기관 비교 AI 브리핑에서 예상하지 못한 오류가 났을 때 보여 줄 문구.
# 내부 정보가 새지 않도록 원본 오류 내용은 담지 않습니다.
INSTITUTION_BRIEFING_UNEXPECTED_ERROR = (
    "기관 비교 AI 브리핑을 만드는 중 예상하지 못한 문제가 발생했습니다. "
    "잠시 후 다시 시도해 주세요."
)

# 기관 이름을 지표·탭 제목에 넣을 때 쓰는 최대 길이(글자).
# 이보다 길면 앞쪽 단어만 남깁니다(예: "Berkshire Hathaway" -> "Berkshire").
# 전체 기관명은 비교 화면 위쪽에 따로 표시하므로 정보가 사라지지 않습니다.
INSTITUTION_LABEL_MAX_LENGTH = 16

# 분기 비교 화면에서 탭으로 나눌 변화 구분 순서.
CHANGE_STATUS_ORDER = [
    STATUS_NEW,
    STATUS_INCREASED,
    STATUS_DECREASED,
    STATUS_EXITED,
    STATUS_UNCHANGED,
]

# 변화 차트에 보여 줄 종목 수(절대값 기준 비중 변화 상위).
TOP_WEIGHT_CHANGE_COUNT = 10

# 비중 변화 차트 색. 값이 커진 쪽과 작아진 쪽을 반대 색으로 나타내는 '발산형' 한 쌍이며,
# 국내 관행에 맞춰 확대는 빨강, 축소는 파랑을 씁니다.
COLOR_INCREASE = "#e34948"
COLOR_DECREASE = "#2a78d6"

# 차트의 축·범례 글자와 격자선 색.
# 표를 다크로 만들기 위해 Streamlit 테마 자체를 어둡게 잡았기 때문에,
# 차트가 그 테마를 따라가 흐릿해지지 않도록 여기서 밝은 배경용 색을 못 박습니다.
CHART_LABEL_COLOR = "#000000"
CHART_GRID_COLOR = "rgba(0, 0, 0, 0.12)"

# ---------------------------------------------------------------------------
# 화면 디자인(에디토리얼 컬러 시스템)
#
# 아래 값들은 '보이는 모습'만 정합니다. 계산·조회·상태 관리와는 무관하며,
# 이 블록을 통째로 지워도 분석 기능은 그대로 동작합니다.
# ---------------------------------------------------------------------------

# 사이트 전체 배경(따뜻한 종이색).
COLOR_BACKGROUND = "#D4CFC2"
# 메인 제목과 주요 헤딩(짙은 녹색).
COLOR_HEADING = "#00533E"
# 상세 설명과 일반 본문.
COLOR_BODY = "#000000"
# 주요 CTA(실행 버튼)와 활성 상태.
COLOR_ACCENT = "#F7633D"
# 보조 강조와 일부 테두리.
COLOR_SECONDARY = "#A35D3F"
# 배지, 작은 태그, 포인트 요소.
COLOR_BADGE = "#ECB97A"

# --- 표(다크 테이블) 전용 색 ------------------------------------------------
#
# 사이트 배경은 밝은 베이지지만, 표만은 검은색 기반의 다크 테이블로 둡니다.
#
# 주의: st.dataframe의 열 헤더와 칸은 HTML이 아니라 <canvas>(그림판)에 그려져
# CSS가 닿지 않습니다. 그래서 아래 값들은 `.streamlit/config.toml`의 [theme]에도
# 똑같이 적어 두어야 하며, 두 곳이 어긋났는지는 테스트에서 확인합니다.

# 열 헤더 배경(검정).
COLOR_TABLE_HEADER_BG = "#000000"
# 본문 칸 배경(매우 어두운 차콜).
COLOR_TABLE_BG = "#0F0F0F"
# 본문·헤더 글씨(흰색).
COLOR_TABLE_TEXT = "#FFFFFF"
# 표 안의 보조 텍스트. 어두운 회색 대신 밝은 베이지를 써서 대비를 높입니다.
COLOR_TABLE_MUTED_TEXT = "#D4CFC2"
# 행·열 구분선. #A35D3F를 어두운 표 배경 위에 옅게 얹은 값입니다.
COLOR_TABLE_BORDER = "#593627"
# 마우스를 올렸을 때 행 색. #ECB97A를 아주 옅게 얹은 값입니다.
COLOR_TABLE_HOVER = "#30281F"

# Streamlit이 hover 색을 만들 때 섞는 재료(config.toml의 secondaryBackgroundColor).
# backgroundColor와 30% 섞이면 위 COLOR_TABLE_HOVER가 나옵니다.
COLOR_TABLE_HOVER_SOURCE = "#7D6244"

# 표 글씨 굵기. Streamlit은 열 헤더 굵기를 따로 받지 않고 테마의 기본 글씨
# 굵기(baseFontWeight)를 쓰므로, 헤더를 굵게 하려면 이 값을 올려야 합니다.
TABLE_FONT_WEIGHT = 600

# 표 밖(사이트 본문)에서 되돌릴 보통 굵기와 강조 굵기.
BODY_FONT_WEIGHT = 400
BODY_STRONG_FONT_WEIGHT = 700

# Hero 배경으로 쓸 이미지. 없어도 앱은 그대로 동작합니다(아래 fallback 사용).
HERO_IMAGE_PATH = Path(__file__).parent / "assets" / "wall_street_hero.jpg"

# Hero 배경으로 쓸 이미지 크기(픽셀)와 JPEG 품질.
# 원본이 크면 화면을 다시 그릴 때마다 큰 데이터를 실어 보내게 되므로,
# 가로로 긴 Hero 영역에 필요한 만큼만 잘라 줄여서 씁니다.
# 처리에 실패하면 원본을 그대로 쓰므로 어느 쪽이든 배경은 표시됩니다.
HERO_IMAGE_MAX_WIDTH = 1600
HERO_IMAGE_MAX_HEIGHT = 480
HERO_IMAGE_JPEG_QUALITY = 72

# 세로로 긴 사진을 자를 때 남길 위치(0=위쪽, 1=아래쪽).
# 건물 사이가 잘 보이도록 살짝 위쪽을 남깁니다.
HERO_IMAGE_FOCAL_Y = 0.38

# 이미지 위에 덮는 밝은 베이지 오버레이(검정 오버레이를 쓰지 않습니다).
# 건물 사진은 분위기만 남기고, 글자가 가장 선명하게 보이도록 합니다.
HERO_OVERLAY = (
    "linear-gradient(100deg,"
    " rgba(212, 207, 194, 0.94) 0%,"
    " rgba(212, 207, 194, 0.88) 46%,"
    " rgba(212, 207, 194, 0.78) 100%)"
)

# 이미지의 채도·대비를 낮춰 배경으로 물러나게 합니다.
HERO_IMAGE_FILTER = "grayscale(0.22) saturate(0.55) brightness(1.06)"

# 이미지 파일이 없거나 읽지 못했을 때 쓰는 절제된 그라데이션.
# 지정 컬러(#D4CFC2, #00533E)만으로 구성합니다.
HERO_FALLBACK_BACKGROUND = (
    "linear-gradient(118deg,"
    " #D4CFC2 0%,"
    " rgba(0, 83, 62, 0.18) 52%,"
    " rgba(0, 83, 62, 0.42) 100%)"
)

# Hero 안에 넣을 문구.
HERO_EYEBROW = "SEC 13F INTELLIGENCE PLATFORM"
HERO_TITLE = "AI 13F Portfolio Analysis"
HERO_SUBTITLE = (
    "Analyze institutional filings, portfolio changes, and comparative "
    "positioning with AI-assisted insights."
)
# 영문 보조 설명 아래에 덧붙이는 한 줄 설명.
# 이 프로젝트를 처음 보는 사용자가 '13F'가 무엇인지 바로 알 수 있게 합니다.
HERO_DESCRIPTION = (
    "13F는 미국 기관투자자가 SEC에 분기별로 제출하는 보유 주식 공시입니다."
)

# 기존 'PoC' 표현은 메인 제목이 아니라 작은 보조 텍스트로만 남깁니다.
HERO_NOTE = "SEC EDGAR 13F-HR 공시 기반 PoC"

# AI 브리핑 결과를 감싸는 카드의 제목과 컨테이너 key.
# key를 주면 Streamlit이 그 컨테이너에 `st-key-<key>` 클래스를 붙여 주므로,
# 버전마다 바뀌는 자동 생성 클래스에 기대지 않고 카드 하나만 골라 꾸밀 수 있습니다.
# (위젯 key가 아니라 표시용 컨테이너 key라 화면 상태(session_state)에는 영향이 없습니다.)
AI_BRIEFING_CARD_TITLE = "AI 브리핑"
AI_BRIEFING_CARD_KEY = "ai_briefing_card"
INSTITUTION_AI_BRIEFING_CARD_KEY = "institution_ai_briefing_card"

# 본문이 지나치게 넓어지지 않도록 제한하는 폭(픽셀).
# layout="wide"로 화면은 넓게 쓰되, 표와 글은 읽기 좋은 폭을 유지합니다.
CONTENT_MAX_WIDTH = 1180


def rgb_triplet(hex_color: str) -> str:
    """'#00533E' 같은 값을 CSS rgba()에 넣을 '0, 83, 62' 형태로 바꿉니다."""
    value = hex_color.lstrip("#")
    return ", ".join(str(int(value[index : index + 2], 16)) for index in (0, 2, 4))


def theme_variables_css() -> str:
    """위에서 정한 여섯 가지 컬러를 CSS 변수로 선언합니다.

    색을 한 곳(위 상수)에서만 관리하기 위해, 아래 스타일시트는 색을 직접
    적지 않고 이 변수만 참조합니다.
    """
    return (
        ":root{"
        f"--f13-bg: {COLOR_BACKGROUND};"
        f"--f13-bg-rgb: {rgb_triplet(COLOR_BACKGROUND)};"
        f"--f13-heading: {COLOR_HEADING};"
        f"--f13-heading-rgb: {rgb_triplet(COLOR_HEADING)};"
        f"--f13-body: {COLOR_BODY};"
        f"--f13-accent: {COLOR_ACCENT};"
        f"--f13-accent-rgb: {rgb_triplet(COLOR_ACCENT)};"
        f"--f13-secondary: {COLOR_SECONDARY};"
        f"--f13-secondary-rgb: {rgb_triplet(COLOR_SECONDARY)};"
        f"--f13-badge: {COLOR_BADGE};"
        f"--f13-badge-rgb: {rgb_triplet(COLOR_BADGE)};"
        f"--f13-table-header-bg: {COLOR_TABLE_HEADER_BG};"
        f"--f13-table-bg: {COLOR_TABLE_BG};"
        f"--f13-table-text: {COLOR_TABLE_TEXT};"
        f"--f13-table-muted: {COLOR_TABLE_MUTED_TEXT};"
        f"--f13-table-border: {COLOR_TABLE_BORDER};"
        f"--f13-table-hover: {COLOR_TABLE_HOVER};"
        f"--f13-body-weight: {BODY_FONT_WEIGHT};"
        f"--f13-strong-weight: {BODY_STRONG_FONT_WEIGHT};"
        f"--f13-content-width: {CONTENT_MAX_WIDTH}px;"
        "}"
    )


# 화면 전체에 적용할 스타일시트.
# 색은 위 theme_variables_css()가 선언한 변수만 씁니다.
# Streamlit 내부 클래스 이름은 버전마다 달라지므로 data-testid를 기준으로 잡습니다.
PAGE_STYLE_CSS = """
/* ── 바탕색 통일 ─────────────────────────────────────────────────────── */
/* 상단 영역(stHeader)과 본문(stMain)이 서로 다른 색으로 보이지 않게 합니다.
   테마 자체는 표를 어둡게 그리려고 다크로 잡아 두었으므로(.streamlit/config.toml),
   화면에 보이는 부분은 여기서 전부 밝은 베이지로 다시 칠합니다. */
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stBottom"] > div {
    background-color: var(--f13-bg);
    background-image: none;
}

[data-testid="stHeader"] {
    border-bottom: none;
    box-shadow: none;
}

[data-testid="stToolbar"] {
    background: transparent;
}

/* 화면은 넓게 쓰되, 글과 표는 읽기 좋은 폭 안에 둡니다. */
[data-testid="stMainBlockContainer"] {
    max-width: var(--f13-content-width);
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}

/* ── 글자색 ──────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    color: var(--f13-body);
}

[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li,
[data-testid="stMarkdown"] strong,
[data-testid="stWidgetLabel"] p,
label p {
    color: var(--f13-body);
}

/* 섹션 제목은 짙은 녹색. 한글 가독성을 위해 본문 글꼴(sans-serif)을 유지합니다. */
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stHeading"] h4,
[data-testid="stHeading"] h5,
[data-testid="stHeading"] h6 {
    color: var(--f13-heading);
    letter-spacing: -0.01em;
}

/* 설명 캡션은 본문보다 한 단계 옅은 검정으로 둡니다. */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: rgba(0, 0, 0, 0.72);
}

/* 본문에 섞인 링크와 코드 조각(AI 브리핑 Markdown 등).
   다크 테마 기본색(밝은 파랑·초록)이 베이지 배경에서 흐려지지 않도록 고정합니다. */
[data-testid="stMarkdown"] a,
[data-testid="stAlertContainer"] a {
    color: var(--f13-secondary);
}

[data-testid="stMarkdown"] code,
[data-testid="stAlertContainer"] code {
    background-color: rgba(var(--f13-badge-rgb), 0.35);
    color: var(--f13-body);
}

/* ── 글자 굵기 ───────────────────────────────────────────────────────── */
/* 표(canvas) 헤더를 굵게 그리려면 테마의 baseFontWeight를 올려야 하는데,
   그 값은 화면 전체 글씨에도 함께 걸립니다. 표 밖의 글은 예전처럼 보통
   굵기로 보이도록 여기서 되돌립니다(표 안 글씨는 CSS가 닿지 않아 그대로 굵게). */
body,
[data-testid="stAppViewContainer"],
[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stAlertContainer"],
[data-testid="stAlertContainer"] p,
[data-testid="stAlertContainer"] li,
[data-testid="stWidgetLabel"] p,
label p {
    font-weight: var(--f13-body-weight);
}

[data-testid="stMarkdown"] strong,
[data-testid="stAlertContainer"] strong {
    font-weight: var(--f13-strong-weight);
}

/* 구분선은 금융 저널처럼 얇은 한 줄로. */
[data-testid="stMain"] hr {
    border: none;
    border-top: 1px solid rgba(var(--f13-heading-rgb), 0.25);
    margin: 2rem 0 1.6rem;
}

/* ── 버튼 ────────────────────────────────────────────────────────────── */
/* key, callback, disabled 조건은 파이썬 쪽 그대로이고 색만 바꿉니다. */
/* 보조 버튼: 테두리와 hover에 #A35D3F를 쓰고, 글자는 대비가 높은 검정으로 둡니다.
   (베이지 배경 위 갈색 글씨는 명도 대비가 3.2:1로 낮아 읽기 어렵습니다.) */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background-color: transparent;
    color: var(--f13-body);
    border: 1px solid rgba(var(--f13-secondary-rgb), 0.65);
    border-radius: 2px;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: none;
    transition: background-color 0.15s ease, color 0.15s ease;
}

.stButton > button:hover,
.stButton > button:focus-visible,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:focus-visible {
    background-color: var(--f13-secondary);
    color: #FFFFFF;
    border-color: var(--f13-secondary);
}

/* 주요 실행 버튼(type="primary"). 주황 바탕 위 검정 글씨로 대비를 확보합니다. */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background-color: var(--f13-accent);
    color: var(--f13-body);
    border: 1px solid rgba(var(--f13-secondary-rgb), 0.55);
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus-visible,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:focus-visible {
    background-color: var(--f13-secondary);
    color: #FFFFFF;
    border-color: var(--f13-secondary);
}

.stButton > button:disabled,
.stButton > button:disabled:hover {
    background-color: transparent;
    color: rgba(0, 0, 0, 0.35);
    border-color: rgba(0, 0, 0, 0.2);
}

/* ── 선택 상자 ───────────────────────────────────────────────────────── */
/* 화면 안의 모든 st.selectbox(분석 대상, 공시 조회, 보유 종목, 동일 분기 비교 등)에
   같은 규칙이 걸리도록 위젯 종류(data-testid)로만 잡습니다.
   오른쪽 화살표를 흰색으로 두기 위해, 상자 자체를 짙은 녹색 바탕으로 바꿔
   글자·화살표가 배경과 충분한 대비를 갖게 했습니다. */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: var(--f13-heading);
    border: 1px solid rgba(var(--f13-heading-rgb), 0.9);
    border-radius: 2px;
    color: #FFFFFF;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--f13-accent);
    box-shadow: none;
}

/* 선택된 값과 placeholder(값을 고르기 전 안내 글) 모두 흰색으로 둡니다. */
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] input {
    color: #FFFFFF;
    -webkit-text-fill-color: #FFFFFF;
}

/* 오른쪽 드롭다운 화살표(BaseWeb select 안의 svg 아이콘).
   아이콘이 currentColor를 쓰는 경우와 fill을 직접 쓰는 경우가 모두 있어
   color와 fill을 함께 지정합니다. */
[data-testid="stSelectbox"] svg,
[data-testid="stSelectbox"] div[data-baseweb="select"] svg,
[data-testid="stSelectbox"] div[data-baseweb="select"] svg *,
[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="icon"] svg,
[data-testid="stSelectbox"] div[data-baseweb="select"] [aria-hidden="true"] svg {
    color: #FFFFFF;
    fill: #FFFFFF;
}

/* 펼쳐진 목록은 화면 밖(포털)에 그려지므로 따로 색을 맞춰 줍니다. */
div[data-baseweb="popover"] ul[role="listbox"] {
    background-color: var(--f13-bg);
    border: 1px solid rgba(var(--f13-heading-rgb), 0.3);
}

div[data-baseweb="popover"] li[role="option"] {
    background-color: transparent;
    color: var(--f13-body);
}

div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background-color: rgba(var(--f13-badge-rgb), 0.55);
    color: var(--f13-body);
}

/* ── 탭 ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid rgba(var(--f13-heading-rgb), 0.25);
    gap: 0.25rem;
}

[data-testid="stTab"] {
    color: rgba(0, 0, 0, 0.68);
    font-weight: 600;
}

[data-testid="stTab"]:hover {
    color: var(--f13-secondary);
}

[data-testid="stTab"][aria-selected="true"] {
    color: var(--f13-heading);
    background-color: rgba(var(--f13-badge-rgb), 0.32);
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--f13-accent);
}

[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: transparent;
}

/* ── 지표 카드 ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: rgba(var(--f13-badge-rgb), 0.14);
    border: 1px solid rgba(var(--f13-heading-rgb), 0.2);
    border-left: 3px solid var(--f13-badge);
    border-radius: 2px;
    padding: 0.85rem 1rem;
}

[data-testid="stMetricLabel"] p {
    color: rgba(0, 0, 0, 0.75);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--f13-heading);
}

/* ── 표(검은색 기반 다크 테이블) ─────────────────────────────────────── */
/* 사이트 배경은 밝은 베이지지만, 표만은 검정 기반 다크 테이블로 둡니다.
   st.dataframe의 열 헤더와 칸은 <canvas>(그림판)에 그려져 CSS가 닿지 않으므로,
   그 색은 .streamlit/config.toml의 [theme]에 같은 값으로 적어 두었습니다.
   여기서는 표를 감싸는 상자, 도구 모음, 그리고 HTML로 그려지는 표를 맞춥니다. */
[data-testid="stDataFrame"] {
    background-color: var(--f13-table-bg);
    border: 1px solid var(--f13-table-border);
    border-radius: 2px;
    overflow: hidden;
}

/* 표 위에 떠 있는 도구 모음(검색·다운로드·전체화면)도 어두운 표에 맞춥니다.
   밝은 배경에 검은 아이콘을 두면 어두운 표 위에서 보이지 않게 됩니다. */
[data-testid="stElementToolbar"] {
    background-color: var(--f13-table-header-bg);
    border: 1px solid var(--f13-table-border);
    border-radius: 2px;
}

[data-testid="stElementToolbar"] button,
[data-testid="stElementToolbar"] button svg,
[data-testid="stElementToolbar"] button svg * {
    color: var(--f13-table-text);
    fill: var(--f13-table-text);
}

[data-testid="stElementToolbar"] button:hover {
    background-color: rgba(var(--f13-badge-rgb), 0.22);
}

/* st.table과 Markdown 표(AI 브리핑이 표를 만들어 낼 때)에는 CSS가 그대로
   적용되므로, st.dataframe과 같은 다크 테이블로 맞춥니다. */
[data-testid="stTable"] table,
[data-testid="stMarkdown"] table {
    border-collapse: collapse;
    background-color: var(--f13-table-bg);
    color: var(--f13-table-text);
    overflow: hidden;
}

[data-testid="stTable"] thead th,
[data-testid="stMarkdown"] thead th {
    background-color: var(--f13-table-header-bg);
    color: var(--f13-table-text);
    font-weight: 600;
    border-bottom: 1px solid var(--f13-table-border);
    border-right: 1px solid var(--f13-table-border);
}

[data-testid="stTable"] tbody td,
[data-testid="stTable"] tbody th,
[data-testid="stMarkdown"] tbody td,
[data-testid="stMarkdown"] tbody th {
    background-color: var(--f13-table-bg);
    color: var(--f13-table-text);
    border-bottom: 1px solid var(--f13-table-border);
    border-right: 1px solid var(--f13-table-border);
}

/* 표 안의 보조 텍스트는 어두운 회색 대신 밝은 베이지로 두어 대비를 높입니다. */
[data-testid="stTable"] tbody th,
[data-testid="stMarkdown"] tbody th,
[data-testid="stTable"] caption,
[data-testid="stMarkdown"] caption {
    color: var(--f13-table-muted);
}

[data-testid="stTable"] tbody tr:hover td,
[data-testid="stTable"] tbody tr:hover th,
[data-testid="stMarkdown"] tbody tr:hover td,
[data-testid="stMarkdown"] tbody tr:hover th {
    background-color: var(--f13-table-hover);
}

/* ── AI 브리핑 카드 ──────────────────────────────────────────────────── */
/* st.container(border=True, key=...)가 붙여 주는 `st-key-<key>` 클래스로
   해당 컨테이너 하나만 골라 꾸밉니다. 단일 기관 브리핑과 기관 비교 브리핑이
   같은 규칙을 공유하므로 두 카드의 모양이 항상 같습니다. */
.st-key-ai_briefing_card,
.st-key-institution_ai_briefing_card {
    background-color: rgba(var(--f13-badge-rgb), 0.2);
    border: 1px solid var(--f13-secondary);
    border-radius: 2px;
    box-shadow: none;
    padding: 1.3rem 1.5rem 1.1rem;
    margin: 0.4rem 0 0.6rem;
}

.f13-briefing-card__title {
    margin: 0 0 0.9rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(var(--f13-secondary-rgb), 0.45);
    color: var(--f13-heading);
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}

/* 카드 안의 본문은 검정으로, 문단 사이는 조금 넉넉하게 둡니다.
   (AI 응답의 Markdown 형식은 그대로 두고 색과 여백만 정합니다.) */
.st-key-ai_briefing_card [data-testid="stMarkdown"] p,
.st-key-ai_briefing_card [data-testid="stMarkdown"] li,
.st-key-ai_briefing_card [data-testid="stMarkdown"] strong,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] p,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] li,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] strong {
    color: var(--f13-body);
    line-height: 1.75;
}

.st-key-ai_briefing_card [data-testid="stMarkdown"] p,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] p {
    margin-bottom: 0.85rem;
}

.st-key-ai_briefing_card [data-testid="stMarkdown"] h1,
.st-key-ai_briefing_card [data-testid="stMarkdown"] h2,
.st-key-ai_briefing_card [data-testid="stMarkdown"] h3,
.st-key-ai_briefing_card [data-testid="stMarkdown"] h4,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] h1,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] h2,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] h3,
.st-key-institution_ai_briefing_card [data-testid="stMarkdown"] h4 {
    color: var(--f13-heading);
    margin-top: 1.1rem;
}

/* ── 안내·경고·오류 상자 ─────────────────────────────────────────────── */
/* Streamlit 기본색(파랑 등) 대신 지정 컬러만 쓰되, 종류별 구분은 유지합니다. */
[data-testid="stAlertContainer"] {
    border-radius: 2px;
    color: var(--f13-body);
}

[data-testid="stAlertContainer"] p,
[data-testid="stAlertContainer"] li,
[data-testid="stAlertContainer"] strong {
    color: var(--f13-body);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    background-color: rgba(var(--f13-secondary-rgb), 0.1);
    border-left: 4px solid var(--f13-secondary);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    background-color: rgba(var(--f13-heading-rgb), 0.12);
    border-left: 4px solid var(--f13-heading);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    background-color: rgba(var(--f13-badge-rgb), 0.35);
    border-left: 4px solid var(--f13-badge);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    background-color: rgba(var(--f13-accent-rgb), 0.16);
    border-left: 4px solid var(--f13-accent);
}

/* ── 진행 표시와 상단 도구 ───────────────────────────────────────────── */
/* 테마가 다크라 기본 글자·아이콘이 흰색으로 나옵니다. 베이지 배경 위에서
   보이지 않는 일이 없도록 본문 색으로 되돌립니다. */
[data-testid="stSpinner"],
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span,
[data-testid="stSpinner"] svg,
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] label,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg,
[data-testid="stMainMenu"] svg {
    color: var(--f13-body);
    fill: currentColor;
}

/* ── Hero 영역 ───────────────────────────────────────────────────────── */
.f13-hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    min-height: 360px;
    display: flex;
    align-items: center;
    margin: 0 0 2.2rem;
    padding: 3rem clamp(1.5rem, 4vw, 3.5rem);
    border: 1px solid rgba(var(--f13-heading-rgb), 0.28);
    border-radius: 2px;
}

/* 배경 이미지(또는 fallback 그라데이션). Hero 안에서만 씁니다. */
.f13-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background-position: center 38%;
    background-repeat: no-repeat;
    background-size: cover;
}

/* 이미지 위에 덮는 밝은 베이지 오버레이. 글자가 가장 선명하게 보이도록 합니다. */
.f13-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 1;
    background: var(--f13-hero-overlay);
}

.f13-hero__inner {
    position: relative;
    z-index: 2;
    max-width: 44rem;
}

.f13-hero__eyebrow {
    margin: 0 0 1rem;
    color: var(--f13-secondary);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

.f13-hero__title {
    margin: 0 0 1.1rem;
    color: var(--f13-heading);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.4rem, 5vw, 4.25rem);
    font-weight: 700;
    line-height: 1.04;
    letter-spacing: -0.02em;
}

.f13-hero__subtitle {
    margin: 0 0 1.4rem;
    max-width: 34rem;
    color: var(--f13-body);
    font-size: clamp(0.98rem, 1.2vw, 1.12rem);
    line-height: 1.6;
}

/* 영문 보조 설명 아래 한 줄 설명. 강조하지 않고 보조 설명보다 작게 둡니다. */
.f13-hero__desc {
    margin: 0 0 1.4rem;
    max-width: 34rem;
    color: var(--f13-body);
    font-size: 0.86rem;
    line-height: 1.55;
}

.f13-hero__note {
    display: inline-block;
    margin: 0;
    padding: 0.34rem 0.7rem;
    background-color: var(--f13-badge);
    color: var(--f13-body);
    border-radius: 2px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
"""


@st.cache_data(show_spinner=False)
def hero_image_data_uri(image_path: str) -> str:
    """Hero 배경 이미지를 읽어 CSS에 바로 넣을 수 있는 data URI로 만듭니다.

    외부 주소를 쓰지 않고 저장소 안의 파일만 읽습니다. 파일이 없거나 읽지
    못하면 빈 문자열을 돌려주어, 부르는 쪽이 fallback 배경을 쓰게 합니다.
    (읽기에 실패해도 예외를 밖으로 내보내지 않으므로 앱이 멈추지 않습니다.)
    """
    try:
        raw_image = Path(image_path).read_bytes()
    except OSError:
        return ""

    if not raw_image:
        return ""

    return "data:image/jpeg;base64," + base64.b64encode(
        compact_hero_image(raw_image)
    ).decode("ascii")


def compact_hero_image(raw_image: bytes) -> bytes:
    """가로로 긴 Hero 영역에 필요한 만큼만 이미지를 잘라 줄입니다.

    원본이 크면 화면을 다시 그릴 때마다 큰 데이터를 함께 보내게 되어 느려집니다.
    줄이는 데 실패하면(라이브러리가 없거나 형식이 달라도) 원본을 그대로 씁니다.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(raw_image)) as image:
            hero_ratio = HERO_IMAGE_MAX_WIDTH / HERO_IMAGE_MAX_HEIGHT
            width, height = image.size

            if width / height > hero_ratio:
                # 가로가 더 긴 사진: 좌우를 잘라 냅니다(가운데 기준).
                kept_width = max(1, round(height * hero_ratio))
                left = round((width - kept_width) / 2)
                box = (left, 0, left + kept_width, height)
            else:
                # 세로가 더 긴 사진: 위아래를 잘라 냅니다(위쪽을 조금 더 남김).
                kept_height = max(1, round(width / hero_ratio))
                top = round((height - kept_height) * HERO_IMAGE_FOCAL_Y)
                box = (0, top, width, top + kept_height)

            cropped = image.convert("RGB").crop(box)
            resized = cropped.resize((HERO_IMAGE_MAX_WIDTH, HERO_IMAGE_MAX_HEIGHT))

            buffer = BytesIO()
            resized.save(
                buffer,
                format="JPEG",
                quality=HERO_IMAGE_JPEG_QUALITY,
                optimize=True,
            )
            return buffer.getvalue()
    except Exception:
        # 이미지 처리에 실패해도 배경은 보여야 하므로 원본을 그대로 돌려줍니다.
        return raw_image


def hero_background_css(image_path=None) -> str:
    """Hero의 배경 이미지(또는 fallback)를 정하는 CSS 조각을 만듭니다.

    이미지가 있으면 채도·대비를 낮춰 배경으로 물러나게 하고,
    없으면 #D4CFC2와 #00533E만 쓴 절제된 그라데이션을 씁니다.
    """
    data_uri = hero_image_data_uri(str(image_path or HERO_IMAGE_PATH))

    if not data_uri:
        return (
            ".f13-hero::before{"
            f"background-image: {HERO_FALLBACK_BACKGROUND};"
            "filter: none;"
            "}"
        )

    return (
        ".f13-hero::before{"
        f'background-image: url("{data_uri}");'
        f"filter: {HERO_IMAGE_FILTER};"
        "}"
    )


def hero_overlay_css() -> str:
    """이미지 위에 덮을 밝은 베이지 오버레이를 CSS 변수로 넘깁니다."""
    return f":root{{--f13-hero-overlay: {HERO_OVERLAY};}}"


def hero_html() -> str:
    """Hero 영역의 HTML을 만듭니다(문구는 위 상수 그대로)."""
    return (
        '<div class="f13-hero">'
        '<div class="f13-hero__inner">'
        f'<p class="f13-hero__eyebrow">{HERO_EYEBROW}</p>'
        f'<h1 class="f13-hero__title">{HERO_TITLE}</h1>'
        f'<p class="f13-hero__subtitle">{HERO_SUBTITLE}</p>'
        f'<p class="f13-hero__desc">{HERO_DESCRIPTION}</p>'
        f'<p class="f13-hero__note">{HERO_NOTE}</p>'
        "</div>"
        "</div>"
    )


def apply_page_style(image_path=None) -> None:
    """컬러 시스템과 Hero 배경 스타일을 화면에 적용합니다(표시 전용)."""
    st.markdown(
        "<style>"
        + theme_variables_css()
        + hero_overlay_css()
        + PAGE_STYLE_CSS
        + hero_background_css(image_path)
        + "</style>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """페이지 최상단 Hero 영역을 그립니다(기존 st.title을 대신합니다)."""
    st.markdown(hero_html(), unsafe_allow_html=True)


def render_ai_briefing_card(briefing_text: str, container_key: str) -> None:
    """AI 브리핑 결과를 테두리가 있는 카드 하나로 묶어 보여 줍니다(표시 전용).

    브리핑의 시작과 끝이 눈에 보이도록 st.container(border=True) 안에 제목과
    본문을 함께 넣습니다. 본문은 st.markdown으로 그대로 그리므로 Gemini가
    만든 Markdown 형식(제목, 목록, 굵은 글씨)이 그대로 유지됩니다.

    Args:
        briefing_text: 화면 상태에 담아 둔 AI 브리핑 본문(Markdown).
        container_key: 카드에 붙일 컨테이너 key. 이 값으로 CSS 클래스
            `st-key-<container_key>`가 만들어져 카드 모양이 정해집니다.
    """
    with st.container(border=True, key=container_key):
        st.markdown(
            f'<p class="f13-briefing-card__title">{AI_BRIEFING_CARD_TITLE}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(briefing_text)


@st.cache_data
def load_managers() -> pd.DataFrame:
    """분석 대상 운용사 목록(data/managers.csv)을 읽어옵니다.

    cik은 앞자리 0이 사라지지 않도록 문자열로 읽습니다.
    """
    return pd.read_csv(MANAGERS_CSV, dtype={"cik": str})


def find_manager(managers: pd.DataFrame, name: str) -> dict:
    """운용사 목록에서 이름으로 한 곳을 찾아 name/cik을 돌려줍니다."""
    matched = managers[managers["name"] == name]
    if matched.empty:
        raise LookupError(f"'{name}'을(를) 운용사 목록에서 찾을 수 없습니다.")

    row = matched.iloc[0]
    return {"name": row["name"], "cik": row["cik"]}


def manager_options(managers: pd.DataFrame) -> list[str]:
    """선택 상자에 보여 줄 운용사 이름 목록을 만듭니다.

    목록이 비어 있으면 고를 대상이 없으므로 LookupError를 냅니다.
    """
    names = [str(name) for name in managers["name"].tolist()]
    if not names:
        raise LookupError("운용사 목록이 비어 있습니다.")
    return names


def default_manager_index(names: list[str]) -> int:
    """기본으로 선택할 운용사의 위치를 돌려줍니다.

    기본 운용사가 목록에 없으면 첫 번째 항목을 씁니다.
    """
    if DEFAULT_MANAGER in names:
        return names.index(DEFAULT_MANAGER)
    return 0


def reset_analysis_state(state) -> None:
    """이전 기관의 조회·분석 결과를 화면 상태에서 지웁니다.

    기관을 바꿨을 때만 호출합니다. 같은 기관에서 화면이 다시 그려질 때는
    호출하지 않으므로, 이미 조회한 결과가 불필요하게 사라지지 않습니다.
    """
    for key in ANALYSIS_STATE_KEYS:
        state[key] = None


def sync_selected_manager(state, selected_name: str) -> bool:
    """지금 고른 기관이 직전과 다른지 확인하고, 달라졌으면 결과를 지웁니다.

    화면은 버튼을 누를 때마다 처음부터 다시 실행되므로, '직전에 고른 기관 이름'을
    따로 기억해 두고 그 값과 비교합니다. 같은 기관이면 아무것도 지우지 않습니다.

    Returns:
        기관이 바뀌어서 결과를 지웠으면 True, 같은 기관이라 그대로 두었으면 False.
    """
    if state.get("active_manager_name") == selected_name:
        return False

    reset_analysis_state(state)
    state["active_manager_name"] = selected_name
    return True


def read_sec_user_agent() -> str:
    """SEC 요청에 쓸 User-Agent를 secrets에서 읽어옵니다.

    값 자체는 화면이나 로그에 절대 출력하지 않고, 조회 함수에만 전달합니다.
    """
    user_agent = st.secrets.get("SEC_USER_AGENT", "")
    if not user_agent or not str(user_agent).strip():
        raise LookupError(
            "SEC 접속에 필요한 설정(SEC_USER_AGENT)이 등록되어 있지 않습니다. "
            "`.streamlit/secrets.toml`에 SEC_USER_AGENT 항목을 추가한 뒤 "
            "앱을 다시 실행해 주세요."
        )
    return str(user_agent)


# ---------------------------------------------------------------------------
# SEC 조회 캐시
#
# 같은 기관·같은 공시를 다시 볼 때마다 SEC에 또 요청하면 느리고, SEC의 요청 빈도
# 제한에도 걸릴 수 있습니다. 그래서 화면 쪽에서 조회 결과를 잠시 보관해 두고
# 재사용합니다(캐시). 수집 로직 자체는 services/sec_client.py에 그대로 두고,
# Streamlit 전용 기능인 캐시만 이 파일에서 감쌉니다.
#
# st.cache_data는 함수에 넘긴 인자를 '보관함 이름표'로 씁니다. 따라서 아래
# 두 함수는 cik / limit / accession_number / user_agent가 모두 같을 때만
# 보관해 둔 값을 돌려주고, 하나라도 다르면 SEC에 새로 요청합니다.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=SEC_CACHE_TTL_SECONDS, show_spinner=False)
def cached_recent_filings(cik: str, limit: int, user_agent: str) -> list[dict]:
    """최근 13F 공시 목록을 조회하고 결과를 캐시에 보관합니다.

    반환값은 get_recent_13f_filings와 같은 dict 목록입니다.
    (캐시 이름표: cik, limit, user_agent)
    """
    return get_recent_13f_filings(cik, user_agent=user_agent, limit=limit)


@st.cache_data(ttl=SEC_CACHE_TTL_SECONDS, show_spinner=False)
def cached_13f_holdings(
    cik: str, accession_number: str, user_agent: str
) -> list[dict]:
    """특정 공시의 보유 종목 목록을 조회하고 결과를 캐시에 보관합니다.

    보유 종목 XML은 기관에 따라 수백~수천 줄이라 가장 무거운 요청입니다.
    같은 cik·accession_number를 다시 요청하면 SEC XML을 새로 받지 않습니다.

    반환값은 get_13f_holdings와 같은 holdings 목록입니다.
    (캐시 이름표: cik, accession_number, user_agent)
    """
    return get_13f_holdings(cik, accession_number, user_agent=user_agent)


def clear_sec_caches() -> None:
    """위에서 만든 두 SEC 조회 캐시만 비웁니다.

    st.cache_data.clear()로 앱 전체 캐시를 지우지 않습니다. 그래서 운용사 목록
    (load_managers)처럼 SEC와 무관한 캐시는 그대로 남습니다.
    """
    cached_recent_filings.clear()
    cached_13f_holdings.clear()


def read_gemini_settings() -> tuple[str, str]:
    """Gemini 호출에 쓸 API 키와 모델명을 secrets에서 읽어옵니다.

    값 자체는 화면이나 터미널에 절대 출력하지 않고, 생성 함수에만 전달합니다.
    """
    api_key = str(st.secrets.get("GEMINI_API_KEY", "") or "").strip()
    model_name = str(st.secrets.get("GEMINI_MODEL", "") or "").strip()

    missing = []
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if not model_name:
        missing.append("GEMINI_MODEL")

    if missing:
        raise LookupError(
            f"AI 브리핑에 필요한 설정({', '.join(missing)})이 등록되어 있지 않습니다. "
            "`.streamlit/secrets.toml`에 해당 항목을 추가한 뒤 앱을 다시 실행해 주세요."
        )

    return api_key, model_name


def to_display_table(filings: list[dict]) -> pd.DataFrame:
    """조회 결과를 화면에 표시할 표 형태로 정리합니다."""
    table = pd.DataFrame(filings, columns=list(COLUMN_LABELS))
    return table.rename(columns=COLUMN_LABELS)


def format_filing_label(filing: dict) -> str:
    """selectbox에 보여 줄 공시 설명을 만듭니다.

    사용자가 어떤 분기의 공시인지 알 수 있도록 제출일과 기준일을 함께 씁니다.
    """
    filing_date = filing.get("filing_date") or "제출일 미확인"
    report_date = filing.get("report_date") or "기준일 미확인"
    return f"제출일 {filing_date} · 기준일(분기 말) {report_date}"


def build_holdings_table(holdings: list[dict]) -> pd.DataFrame:
    """보유 종목 목록을 화면용 표로 정리합니다.

    하는 일:
        1) 정해진 열 순서로 표를 만듭니다.
        2) 금액과 수량을 숫자로 바꿉니다(바꿀 수 없는 값은 빈칸 처리).
        3) 각 종목의 평가금액을 전체 합계로 나눠 포트폴리오 비중(%)을 계산합니다.
        4) 평가금액이 큰 종목부터 정렬합니다.

    보유 종목이 없거나 금액이 비어 있어도 오류 없이 빈 표를 돌려줍니다.
    """
    table = pd.DataFrame(holdings, columns=HOLDINGS_COLUMNS)

    for column in NUMERIC_HOLDINGS_COLUMNS:
        # 숫자로 바꿀 수 없는 값은 빈 값(NaN)으로 두어 합계 계산에서 제외합니다.
        table[column] = pd.to_numeric(table[column], errors="coerce")

    for column in TEXT_HOLDINGS_COLUMNS:
        table[column] = table[column].fillna("").astype(str)

    total_value = sum_portfolio_value(table)
    if total_value > 0:
        table["portfolio_weight"] = table["reported_value"] / total_value * 100
    else:
        # 금액이 모두 비어 있으면 비중을 계산할 수 없으므로 빈칸으로 둡니다.
        table["portfolio_weight"] = pd.NA

    sorted_table = table.sort_values(
        "reported_value", ascending=False, na_position="last"
    )
    return sorted_table.reset_index(drop=True)


def sum_portfolio_value(table: pd.DataFrame) -> float:
    """보유 종목 표의 공시 평가금액 합계를 돌려줍니다. 값이 없으면 0입니다."""
    if table.empty:
        return 0.0

    total = table["reported_value"].sum(skipna=True)
    return float(total) if pd.notna(total) else 0.0


def format_percent(value, digits: int = 2) -> str:
    """비율을 부호가 붙은 문자열로 바꿉니다. 계산 불가한 값은 안내 문구로 바꿉니다."""
    if value is None or pd.isna(value):
        return "계산 불가"
    return f"{value:+,.{digits}f}%"


def table_height(row_count: int) -> int | str:
    """표에 지정할 높이를 정합니다.

    보유 종목이 많은 기관(예: Bridgewater)은 표가 화면을 끝없이 늘리지 않도록
    높이(양의 정수, 픽셀)를 정해 표 안에서 스크롤되게 합니다. 행이 적으면
    "content"를 돌려주어 내용에 맞춘 높이를 쓰게 합니다.

    Streamlit이 허용하는 값은 양의 정수, "content", "stretch"뿐이므로 어떤
    입력에서도 None을 돌려주지 않습니다. 행 개수가 숫자가 아니거나 비어 있어도
    "content"로 안전하게 처리합니다.

    어느 쪽이든 행을 잘라내지 않으므로 전체 데이터는 그대로 유지됩니다.
    """
    try:
        rows = int(row_count)
    except (TypeError, ValueError):
        return AUTO_TABLE_HEIGHT

    if rows > LARGE_TABLE_ROW_THRESHOLD:
        return LARGE_TABLE_HEIGHT
    return AUTO_TABLE_HEIGHT


def comparison_column_config() -> dict:
    """분기 비교 표의 숫자 표시 형식을 정합니다.

    열 이름은 분석 결과 그대로 두고, 값의 출처 설명은 도움말(?)과 캡션으로 알려 줍니다.
    """
    return {
        "put_call": st.column_config.TextColumn(
            help="옵션 구분. 값이 있으면 Put/Call 보유이고, 비어 있으면 일반 주식 보유입니다."
        ),
        "share_type": st.column_config.TextColumn(
            help="수량 단위 (SH=주식 수, PRN=원금액)"
        ),
        "previous_reported_value": st.column_config.NumberColumn(
            help="이전 분기 공시 평가금액 (SEC Information Table의 reported value 필드)",
            format="localized",
        ),
        "current_reported_value": st.column_config.NumberColumn(
            help="현재 분기 공시 평가금액 (SEC Information Table의 reported value 필드)",
            format="localized",
        ),
        "reported_value_change": st.column_config.NumberColumn(
            help="공시 평가금액 증감 (SEC Information Table의 reported value 필드 기준)",
            format="localized",
        ),
        "previous_shares": st.column_config.NumberColumn(
            help="이전 분기 보유수량", format="localized"
        ),
        "current_shares": st.column_config.NumberColumn(
            help="현재 분기 보유수량", format="localized"
        ),
        "shares_change": st.column_config.NumberColumn(
            help="보유수량 증감", format="localized"
        ),
        "previous_weight": st.column_config.NumberColumn(
            help="이전 분기 포트폴리오 비중 (%)", format="%.2f%%"
        ),
        "current_weight": st.column_config.NumberColumn(
            help="현재 분기 포트폴리오 비중 (%)", format="%.2f%%"
        ),
        "weight_change_pct_point": st.column_config.NumberColumn(
            help="비중 변화 (%포인트)", format="%+.2f"
        ),
    }


# ---------------------------------------------------------------------------
# 기관 간 동일 분기 비교 도우미
#
# 계산은 services/institution_comparison.py가 모두 맡고, 아래 함수들은 그 결과를
# 화면에 어떻게 보여 줄지(기본 선택값, 상태 초기화, 표시 형식)만 정합니다.
# ---------------------------------------------------------------------------


def institution_default_index(names: list[str], preferred: str, avoid: str = "") -> int:
    """비교 화면 선택 상자의 기본 선택 위치를 돌려줍니다.

    기본 기관이 목록에 없으면, 반대쪽 기본 기관(avoid)과 다른 첫 번째 기관을 고릅니다.
    두 선택 상자가 처음부터 같은 기관을 가리키지 않게 하기 위함입니다.
    """
    if preferred in names:
        return names.index(preferred)

    for index, name in enumerate(names):
        if name != avoid:
            return index

    return 0


def can_compare_institutions(left_name, right_name) -> bool:
    """두 기관을 비교할 수 있는 조합인지 확인합니다.

    같은 기관끼리는 비교할 내용이 없고(모든 종목이 공통 보유가 됩니다),
    기관을 고르지 않은 상태에서도 비교를 실행하지 않습니다.
    """
    left = str(left_name or "").strip()
    right = str(right_name or "").strip()

    return bool(left) and bool(right) and left != right


def reset_institution_comparison_state(state) -> None:
    """기관 간 비교 화면의 결과만 지웁니다.

    지우는 대상은 INSTITUTION_COMPARISON_STATE_KEYS뿐이므로, 위쪽 단일 기관
    분석 결과(공시 목록, 보유 종목, 두 분기 비교, AI 브리핑)는 그대로 남습니다.
    """
    for key in INSTITUTION_COMPARISON_STATE_KEYS:
        state[key] = None


def reset_institution_ai_briefing_state(state) -> None:
    """기관 비교 AI 브리핑 결과만 지웁니다.

    지우는 대상은 INSTITUTION_AI_BRIEFING_STATE_KEYS뿐이므로, 비교 표와 요약
    지표는 화면에 그대로 남고 단일 기관 브리핑(ai_briefing)도 건드리지 않습니다.
    """
    for key in INSTITUTION_AI_BRIEFING_STATE_KEYS:
        state[key] = None


def sync_institution_briefing_report_date(state, report_date) -> bool:
    """비교할 기준일이 직전과 달라졌으면 기관 비교 AI 브리핑만 지웁니다.

    브리핑은 특정 분기의 비교 결과를 설명한 글이므로, 사용자가 기준일을 바꾸면
    다른 분기의 설명이 화면에 남아 있지 않게 합니다. 비교 결과 자체는 지우지
    않습니다(다시 비교하기 전까지 앞서 만든 표를 계속 볼 수 있게 하기 위함입니다).

    Returns:
        기준일이 바뀌어서 브리핑을 지웠으면 True, 같은 기준일이라 그대로 두었으면 False.
    """
    if state.get("active_institution_briefing_date") == report_date:
        return False

    reset_institution_ai_briefing_state(state)
    state["active_institution_briefing_date"] = report_date
    return True


def institution_briefing_payload_from_state(state) -> dict:
    """화면 상태에 담아 둔 기관 비교 결과로 AI 입력 데이터를 만듭니다.

    계산과 상위 종목 선별은 institution_comparison 모듈이 맡고, 이 함수는 화면
    상태에서 필요한 값만 꺼내 전달합니다. 원본 XML이나 전체 보유 종목 목록은
    꺼내지 않습니다.
    """
    return build_institution_comparison_briefing_payload(
        state.get("institution_comparison"),
        state.get("institution_comparison_summary"),
        report_date=state.get("institution_selected_report_date") or "",
        left_manager_name=state.get("institution_comparison_left_name") or "기관 A",
        right_manager_name=state.get("institution_comparison_right_name") or "기관 B",
    )


def run_institution_ai_briefing(state) -> None:
    """기관 비교 AI 브리핑을 만들어 기관 비교 전용 화면 상태에 담습니다.

    버튼을 눌렀을 때만 호출합니다. 화면이 다시 그려지는 것만으로는 호출되지
    않으므로 Gemini API가 반복 호출되지 않습니다.

    Gemini 호출과 오류 문구 변환은 단일 기관 브리핑과 같은
    llm_client.generate_briefing을 그대로 씁니다. 성공하면
    institution_ai_briefing에, 실패하면 institution_ai_briefing_error에 담고
    예외를 밖으로 내보내지 않습니다(화면이 멈추지 않게 하기 위함입니다).
    """
    reset_institution_ai_briefing_state(state)

    try:
        # API 키와 모델명은 여기서 읽어 생성 함수에만 넘깁니다. 화면에 표시하지 않습니다.
        gemini_api_key, gemini_model = read_gemini_settings()

        prompt = build_institution_comparison_briefing_prompt(
            institution_briefing_payload_from_state(state)
        )

        with st.spinner(INSTITUTION_BRIEFING_SPINNER_TEXT):
            state["institution_ai_briefing"] = generate_briefing(
                prompt,
                api_key=gemini_api_key,
                model_name=gemini_model,
            )
    except LookupError as error:
        # GEMINI_API_KEY 또는 GEMINI_MODEL 설정이 없는 경우.
        state["institution_ai_briefing_error"] = str(error)
    except LlmApiError as error:
        # llm_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
        state["institution_ai_briefing_error"] = str(error)
    except Exception:
        # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
        state["institution_ai_briefing_error"] = INSTITUTION_BRIEFING_UNEXPECTED_ERROR


def sync_institution_pair(state, left_name: str, right_name: str) -> bool:
    """비교할 두 기관이 직전과 달라졌으면 비교 결과를 지웁니다.

    화면은 버튼을 누를 때마다 처음부터 다시 실행되므로, '직전에 고른 두 기관'을
    따로 기억해 두고 그 값과 비교합니다. 같은 조합이면 아무것도 지우지 않습니다.

    Returns:
        기관 조합이 바뀌어서 결과를 지웠으면 True, 그대로 두었으면 False.
    """
    pair = (left_name, right_name)

    if state.get("active_institution_pair") == pair:
        return False

    reset_institution_comparison_state(state)
    state["active_institution_pair"] = pair
    return True


def short_institution_name(
    name, max_length: int = INSTITUTION_LABEL_MAX_LENGTH
) -> str:
    """지표·탭 제목에 넣을 짧은 기관 이름을 만듭니다.

    이름이 길면 최대 길이를 넘지 않는 만큼 앞쪽 단어만 남깁니다
    (예: "Pershing Square Capital Management" -> "Pershing Square").
    전체 기관명은 비교 화면 위쪽에 따로 표시하므로 정보가 사라지지 않습니다.
    """
    text = str(name or "").strip()
    if len(text) <= max_length:
        return text

    shortened = ""
    for word in text.split():
        candidate = f"{shortened} {word}".strip()
        if shortened and len(candidate) > max_length:
            break
        shortened = candidate

    # 첫 단어 하나가 이미 최대 길이보다 길면 글자 수로 잘라 냅니다.
    return shortened or text[:max_length]


def format_overlap_percent(value, digits: int = 1) -> str:
    """중복률·중복도를 화면에 표시할 문자열로 바꿉니다.

    증감이 아니라 '전체 중 몇 %'를 나타내는 값이므로 부호(+)를 붙이지 않습니다.
    계산할 수 없는 값은 안내 문구로 바꿉니다.
    """
    if value is None or pd.isna(value):
        return "계산 불가"
    return f"{float(value):,.{digits}f}%"


def institution_filing_for_date(filings_by_date, report_date) -> dict:
    """기준일로 정리해 둔 공시 목록에서 해당 분기의 공시 한 건을 꺼냅니다.

    같은 기준일에 여러 건이 있을 때 어느 것을 쓸지는
    institution_comparison.index_filings_by_report_date의 기준을 그대로 따릅니다.
    찾지 못하면 빈 딕셔너리를 돌려주어 화면이 멈추지 않게 합니다.
    """
    if not isinstance(filings_by_date, dict) or not report_date:
        return {}

    filing = filings_by_date.get(report_date)
    return filing if isinstance(filing, dict) else {}


def format_institution_filing_summary(label: str, filing: dict) -> str:
    """비교 대상 공시의 접수번호와 제출일을 간단히 보여 줄 문구를 만듭니다."""
    filing = filing or {}
    return (
        f"{label}\n\n"
        f"- 접수번호(accession_number): `{filing.get('accession_number', '-')}`\n"
        f"- 제출일(filing_date): `{filing.get('filing_date', '-')}`"
    )


def institution_rows_by_type(comparison, holding_type: str) -> pd.DataFrame:
    """비교 결과에서 한 가지 보유 유형(공통/기관 A 단독/기관 B 단독)만 골라냅니다.

    결과가 없거나 유형 열이 없어도 오류 없이 빈 표를 돌려줍니다.
    """
    if comparison is None or len(comparison) == 0:
        return pd.DataFrame(columns=INSTITUTION_DISPLAY_COLUMNS)

    if HOLDING_TYPE not in comparison.columns:
        return pd.DataFrame(columns=INSTITUTION_DISPLAY_COLUMNS)

    return comparison[comparison[HOLDING_TYPE] == holding_type]


def institution_column_config() -> dict:
    """기관 간 비교 표의 한국어 열 제목과 숫자 표시 형식을 정합니다.

    열 제목에 기관명을 반복하지 않고 '기관 A / 기관 B'로 표시합니다.
    어느 기관이 A이고 B인지는 표 위쪽에서 전체 이름으로 보여 줍니다.
    """
    return {
        "issuer_name": st.column_config.TextColumn("종목명"),
        "cusip": st.column_config.TextColumn("CUSIP"),
        "put_call": st.column_config.TextColumn(
            "옵션 구분",
            help="값이 있으면 Put/Call 보유이고, 비어 있으면 일반 주식 보유입니다.",
        ),
        "share_type": st.column_config.TextColumn(
            "수량 단위", help="SH=주식 수, PRN=원금액"
        ),
        "left_reported_value": st.column_config.NumberColumn(
            "기관 A 공시 평가금액",
            help="SEC Information Table의 reported value 필드 (환산하지 않은 공시 원문 값)",
            format="localized",
        ),
        "right_reported_value": st.column_config.NumberColumn(
            "기관 B 공시 평가금액",
            help="SEC Information Table의 reported value 필드 (환산하지 않은 공시 원문 값)",
            format="localized",
        ),
        "left_weight_pct": st.column_config.NumberColumn(
            "기관 A 비중", help="기관 A 포트폴리오에서 차지하는 비율(%)", format="%.2f%%"
        ),
        "right_weight_pct": st.column_config.NumberColumn(
            "기관 B 비중", help="기관 B 포트폴리오에서 차지하는 비율(%)", format="%.2f%%"
        ),
        "weight_gap_pct_point": st.column_config.NumberColumn(
            "비중 차이(%p)",
            help="기관 A 비중 - 기관 B 비중 (%포인트). 양수면 기관 A가 더 많이 담은 종목입니다.",
            format="%+.2f",
        ),
    }


def render_institution_holdings_table(rows) -> None:
    """기관 간 비교 결과 표 하나를 화면에 그립니다.

    비어 있으면 표를 그리지 않고 안내 문구만 보여 줍니다. 표를 그릴 때는
    height에 항상 table_height()의 결과(양의 정수 또는 "content")를 넘기므로
    None이 전달되어 생기던 StreamlitInvalidHeightError가 나지 않습니다.
    """
    if rows is None or len(rows) == 0:
        st.info("해당 종목이 없습니다.")
        return

    st.dataframe(
        rows[INSTITUTION_DISPLAY_COLUMNS],
        hide_index=True,
        width="stretch",
        height=table_height(len(rows)),
        column_config=institution_column_config(),
    )


def top_weight_changes(comparison: pd.DataFrame) -> pd.DataFrame:
    """비중 변화가 큰 상위 종목을 골라 차트용 표로 만듭니다.

    '큰 변화'는 늘어난 쪽과 줄어든 쪽을 함께 보기 위해 절대값 기준으로 고릅니다.
    그래서 신규 편입(크게 늘어남)과 전량 매도(크게 줄어듦)도 함께 나타납니다.
    """
    if comparison is None or comparison.empty:
        return pd.DataFrame(
            columns=["issuer_name", "weight_change_pct_point", "change_status", "변화 방향"]
        )

    table = comparison.copy()
    table["절대 변화"] = table["weight_change_pct_point"].abs()

    top = table.sort_values("절대 변화", ascending=False).head(TOP_WEIGHT_CHANGE_COUNT)
    top = top[top["절대 변화"] > 0]

    # 같은 회사의 옵션 보유는 별개의 포지션이므로 이름에 Put/Call 표시를 덧붙입니다.
    labels = top["issuer_name"].where(top["issuer_name"] != "", top["cusip"])
    if "put_call" in top.columns:
        option_marks = top["put_call"].fillna("").astype(str).str.strip()
        labels = labels.where(option_marks == "", labels + " (" + option_marks + ")")

    # 그래도 같은 이름이 겹치면 막대가 합쳐져 보이므로 CUSIP 뒷자리로 구분합니다.
    if labels.duplicated().any():
        labels = labels + " (" + top["cusip"].str[-4:] + ")"

    return pd.DataFrame(
        {
            "issuer_name": labels,
            "weight_change_pct_point": top["weight_change_pct_point"],
            "change_status": top["change_status"],
            "변화 방향": top["weight_change_pct_point"].apply(
                lambda value: "비중 확대" if value > 0 else "비중 축소"
            ),
        }
    )


def weight_change_chart(chart_data: pd.DataFrame) -> alt.Chart:
    """비중 변화 막대 차트를 만듭니다.

    가로 막대로 그리고, 변화가 큰 종목을 위쪽에 둡니다.
    0을 기준으로 오른쪽(확대)과 왼쪽(축소)을 서로 반대 색으로 표시합니다.
    """
    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X(
                "weight_change_pct_point:Q",
                title="비중 변화 (%포인트)",
                axis=alt.Axis(format="+.2f"),
            ),
            y=alt.Y(
                "issuer_name:N",
                title=None,
                sort=alt.EncodingSortField(
                    field="weight_change_pct_point", order="descending"
                ),
            ),
            color=alt.Color(
                "변화 방향:N",
                title=None,
                scale=alt.Scale(
                    domain=["비중 확대", "비중 축소"],
                    range=[COLOR_INCREASE, COLOR_DECREASE],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("issuer_name:N", title="종목"),
                alt.Tooltip("change_status:N", title="변화 구분"),
                alt.Tooltip(
                    "weight_change_pct_point:Q", title="비중 변화(%p)", format="+.2f"
                ),
            ],
        )
    )

    # 0 기준선. 어디까지가 확대이고 축소인지 눈으로 바로 구분되게 합니다.
    zero_line = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#8a8a85", strokeWidth=1)
        .encode(x="zero:Q")
    )

    # 축·범례 색을 못 박습니다(표를 다크로 만들려고 테마를 어둡게 잡았기 때문에,
    # 그냥 두면 차트 글씨가 흰색으로 그려져 베이지 배경에서 보이지 않습니다).
    return (
        (bars + zero_line)
        .properties(height=alt.Step(26))
        .configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=CHART_LABEL_COLOR,
            titleColor=CHART_LABEL_COLOR,
            gridColor=CHART_GRID_COLOR,
            domainColor=CHART_GRID_COLOR,
        )
        .configure_legend(
            labelColor=CHART_LABEL_COLOR,
            titleColor=CHART_LABEL_COLOR,
        )
        .configure_title(color=COLOR_HEADING)
    )


# ---------------------------------------------------------------------------
# 화면 구성
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI 13F 포트폴리오 분석 PoC",
    page_icon="📊",
    layout="wide",
)

# 컬러 시스템과 Hero 스타일 적용. 화면에 보이는 모습만 바꾸며,
# 아래 분석 기능의 실행 순서나 조건은 건드리지 않습니다.
apply_page_style()

# 기존 st.title을 대신하는 Hero 영역(제목이 중복되지 않게 title은 쓰지 않습니다).
render_hero()

st.subheader("프로젝트 설명")
st.write(
    "미국 증권거래위원회(SEC)에 제출되는 13F 공시 데이터를 불러와 "
    "기관투자자의 주식 보유 현황과 분기별 변화를 분석하고, "
    "그 결과를 Gemini API가 자연어로 쉽게 설명해 주는 웹 애플리케이션입니다."
)

st.subheader("분석 대상")

# 운용사 목록 읽기. 파일이 없거나 형식이 다르면 여기서 안내하고 멈춥니다.
try:
    managers = load_managers()
    manager_names = manager_options(managers)
except FileNotFoundError:
    st.error(
        "운용사 목록 파일(data/managers.csv)을 찾을 수 없습니다. "
        "파일이 있는지 확인해 주세요."
    )
    st.stop()
except (LookupError, KeyError, pd.errors.ParserError):
    st.error(
        "운용사 목록 파일(data/managers.csv)을 읽을 수 없습니다. "
        "name, cik 열과 한 곳 이상의 운용사가 있는지 확인해 주세요."
    )
    st.stop()

selected_manager_name = st.selectbox(
    "분석할 기관투자자를 선택하세요",
    options=manager_names,
    index=default_manager_index(manager_names),
    key="selected_manager_name",
)

# 기관을 바꿨을 때만 앞선 기관의 결과를 지웁니다.
# 같은 기관에서 버튼을 눌러 화면이 다시 그려질 때는 결과가 그대로 남습니다.
sync_selected_manager(st.session_state, selected_manager_name)

try:
    manager = find_manager(managers, selected_manager_name)
except (LookupError, KeyError) as error:
    st.error(str(error))
    st.stop()

st.write(f"**{manager['name']}** (CIK {manager['cik']})")

st.divider()

st.subheader(f"최근 13F 공시 {FILING_LIMIT}건")
st.caption(
    "13F-HR은 운용 규모가 큰 기관투자자가 분기마다 제출하는 보유 종목 보고서입니다. "
    "정정 공시(13F-HR/A)는 제외하고 정기 공시만 조회합니다."
)
st.caption(
    f"한 번 불러온 결과는 {SEC_CACHE_TTL_SECONDS // 3600}시간 동안 재사용해 "
    "SEC 요청을 줄입니다. 최신 상태를 바로 확인하려면 아래 "
    "'SEC 데이터 새로고침'을 눌러 주세요."
)

fetch_column, refresh_column = st.columns([1, 1])

if fetch_column.button("최근 13F 공시 조회", type="primary"):
    # 조회 결과와 오류 메시지를 화면 상태에 보관합니다.
    # 다른 버튼을 눌러 화면이 다시 그려져도 표가 사라지지 않게 하기 위함입니다.
    st.session_state["filings"] = None
    st.session_state["error"] = None
    # 공시 목록을 새로 불러오면, 앞서 조회한 보유 종목 결과는 지웁니다.
    st.session_state["holdings"] = None
    st.session_state["holdings_error"] = None
    st.session_state["holdings_filing"] = None

    try:
        # User-Agent는 여기서 읽어 조회 함수에 바로 넘깁니다. 화면에 표시하지 않습니다.
        user_agent = read_sec_user_agent()

        with st.spinner("SEC EDGAR에서 공시 목록을 불러오는 중입니다..."):
            st.session_state["filings"] = cached_recent_filings(
                manager["cik"],
                FILING_LIMIT,
                user_agent,
            )
    except LookupError as error:
        # SEC_USER_AGENT 설정이 없는 경우.
        st.session_state["error"] = str(error)
    except SecApiError as error:
        # sec_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
        st.session_state["error"] = str(error)
    except Exception:
        # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
        st.session_state["error"] = (
            "공시 목록을 불러오는 중 예상하지 못한 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

# 캐시 때문에 옛 데이터가 보일 때 사용자가 직접 다시 받아올 수 있게 하는 버튼입니다.
# 화면에 이미 나와 있는 표나 기관 선택은 건드리지 않고, 보관해 둔 조회 결과만 비웁니다.
if refresh_column.button(
    "SEC 데이터 새로고침",
    help="보관해 둔 SEC 조회 결과를 비웁니다. 기관 선택과 화면의 분석 결과는 그대로 남습니다.",
):
    clear_sec_caches()
    st.info(CACHE_CLEARED_MESSAGE)

if st.session_state.get("error"):
    st.error(st.session_state["error"])
elif st.session_state.get("filings") is not None:
    filings = st.session_state["filings"]
    if filings:
        st.success(f"최근 13F-HR 공시 {len(filings)}건을 불러왔습니다.")
        st.dataframe(to_display_table(filings), hide_index=True)
    else:
        st.warning(
            f"{manager['name']}의 13F-HR 공시를 찾지 못했습니다. "
            "CIK가 올바른지 확인해 주세요."
        )

st.divider()

# ---------------------------------------------------------------------------
# 보유 종목 조회
# ---------------------------------------------------------------------------

st.subheader("보유 종목 조회")
st.caption(
    "위에서 불러온 공시 중 하나를 골라, 그 공시에 담긴 실제 보유 종목 목록"
    "(INFORMATION TABLE)을 조회합니다."
)

available_filings = st.session_state.get("filings") or []

if not available_filings:
    st.info("먼저 위의 '최근 13F 공시 조회' 버튼을 눌러 공시 목록을 불러와 주세요.")
else:
    selected_index = st.selectbox(
        "조회할 공시를 선택하세요",
        options=range(len(available_filings)),
        format_func=lambda index: format_filing_label(available_filings[index]),
        key="selected_filing_index",
    )
    selected_filing = available_filings[selected_index]

    if st.button("보유 종목 조회", type="primary"):
        st.session_state["holdings"] = None
        st.session_state["holdings_error"] = None
        st.session_state["holdings_filing"] = selected_filing

        try:
            # User-Agent는 여기서 읽어 조회 함수에만 넘깁니다. 화면에 표시하지 않습니다.
            user_agent = read_sec_user_agent()

            with st.spinner("SEC EDGAR에서 보유 종목 목록을 불러오는 중입니다..."):
                st.session_state["holdings"] = cached_13f_holdings(
                    manager["cik"],
                    selected_filing["accession_number"],
                    user_agent,
                )
        except LookupError as error:
            # SEC_USER_AGENT 설정이 없는 경우.
            st.session_state["holdings_error"] = str(error)
        except (SecApiError, ValueError) as error:
            # sec_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
            # (보유 종목 파일을 찾지 못한 경우도 여기에 포함됩니다.)
            st.session_state["holdings_error"] = str(error)
        except Exception:
            # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
            st.session_state["holdings_error"] = (
                "보유 종목을 불러오는 중 예상하지 못한 문제가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )

if st.session_state.get("holdings_error"):
    st.error(st.session_state["holdings_error"])
elif st.session_state.get("holdings") is not None:
    holdings = st.session_state["holdings"]
    holdings_filing = st.session_state.get("holdings_filing") or {}

    if not holdings:
        st.warning(
            "이 공시에서는 보유 종목을 찾지 못했습니다. "
            "다른 공시를 선택해 다시 조회해 주세요."
        )
    else:
        holdings_table = build_holdings_table(holdings)
        total_value = sum_portfolio_value(holdings_table)

        st.success(f"조회한 공시: {format_filing_label(holdings_filing)}")

        # 1) 전체 보유 종목 수 / 2) 전체 공시 평가금액 합계
        count_column, value_column = st.columns(2)
        count_column.metric("전체 보유 종목 수", f"{len(holdings_table):,}개")
        value_column.metric(
            "전체 공시 평가금액 합계", f"{total_value:,.0f}"
        )
        st.caption(
            "공시 평가금액(reported_value)은 **SEC Information Table의 reported value "
            "필드를 사용합니다.** 이 화면은 값을 환산하지 않고 공시 원문 그대로 표시하므로, "
            "금액의 단위는 해당 공시 원문을 기준으로 확인해 주세요. "
            "비중(%)은 같은 공시 안에서의 비율이라 단위와 관계없이 그대로 유효합니다."
        )

        # 3) 상위 10개 종목의 포트폴리오 비중
        st.markdown(f"**상위 {TOP_HOLDINGS_COUNT}개 종목 비중**")
        top_holdings = holdings_table.head(TOP_HOLDINGS_COUNT)[
            ["issuer_name", "reported_value", "portfolio_weight"]
        ]
        st.dataframe(
            top_holdings,
            hide_index=True,
            width="stretch",
            column_config={
                "reported_value": st.column_config.NumberColumn(format="localized"),
                "portfolio_weight": st.column_config.ProgressColumn(
                    help="전체 공시 평가금액 합계에서 이 종목이 차지하는 비율(%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

        # 4) 보유 종목 전체 표
        # 종목 수가 많은 기관도 전체를 그대로 담고, 표시 높이만 정해 표 안에서 스크롤됩니다.
        st.markdown("**보유 종목 전체**")
        st.dataframe(
            holdings_table,
            hide_index=True,
            width="stretch",
            height=table_height(len(holdings_table)),
            column_config={
                "reported_value": st.column_config.NumberColumn(format="localized"),
                "shares": st.column_config.NumberColumn(format="localized"),
                "portfolio_weight": st.column_config.NumberColumn(
                    help="전체 공시 평가금액 합계에서 이 종목이 차지하는 비율(%)",
                    format="%.2f%%",
                ),
            },
        )
        st.caption(
            "표는 공시 평가금액이 큰 종목부터 정렬했습니다. "
            "put_call 칸은 옵션 관련 보유일 때만 값이 표시됩니다. "
            "보유 종목이 많은 기관도 전체 목록을 그대로 담고 있으며, "
            "표 안에서 스크롤해 확인할 수 있습니다."
        )

st.divider()

# ---------------------------------------------------------------------------
# 최근 두 분기 포트폴리오 비교
# ---------------------------------------------------------------------------

st.subheader("최근 두 분기 포트폴리오 비교")
st.caption(
    "최근 13F-HR 공시 2건을 자동으로 불러와, 가장 최신 공시를 '현재 분기', "
    "그 이전 공시를 '이전 분기'로 놓고 보유 종목의 변화를 비교합니다."
)

if st.button("두 분기 변화 분석", type="primary"):
    st.session_state["comparison"] = None
    st.session_state["comparison_summary"] = None
    st.session_state["comparison_filings"] = None
    st.session_state["comparison_error"] = None
    st.session_state["comparison_warning"] = None
    # 비교 결과를 새로 만들면, 앞서 생성한 AI 브리핑은 옛 데이터 기준이므로 지웁니다.
    st.session_state["ai_briefing"] = None
    st.session_state["ai_briefing_error"] = None

    try:
        # User-Agent는 여기서 읽어 조회 함수에만 넘깁니다. 화면에 표시하지 않습니다.
        user_agent = read_sec_user_agent()

        with st.spinner("SEC EDGAR에서 최근 공시 2건을 불러오는 중입니다..."):
            recent_filings = cached_recent_filings(
                manager["cik"],
                FILING_LIMIT,
                user_agent,
            )

        if len(recent_filings) < 2:
            # 공시가 1건뿐이면 비교할 대상이 없습니다. 오류가 아니라 안내로 알립니다.
            st.session_state["comparison_warning"] = (
                f"{manager['name']}의 13F-HR 공시가 {len(recent_filings)}건만 조회되어 "
                "두 분기를 비교할 수 없습니다."
            )
        else:
            # 최신순으로 정렬된 목록이므로 첫 번째가 현재 분기, 두 번째가 이전 분기입니다.
            current_filing = recent_filings[0]
            previous_filing = recent_filings[1]

            # 위 '보유 종목 조회'에서 이미 불러온 공시라면 캐시에 있는 값을 그대로 씁니다.
            with st.spinner("현재 분기 보유 종목을 불러오는 중입니다..."):
                current_holdings = cached_13f_holdings(
                    manager["cik"],
                    current_filing["accession_number"],
                    user_agent,
                )

            with st.spinner("이전 분기 보유 종목을 불러오는 중입니다..."):
                previous_holdings = cached_13f_holdings(
                    manager["cik"],
                    previous_filing["accession_number"],
                    user_agent,
                )

            with st.spinner("두 분기의 변화를 분석하는 중입니다..."):
                comparison_result = compare_holdings(
                    previous_holdings, current_holdings
                )
                st.session_state["comparison"] = comparison_result
                st.session_state["comparison_summary"] = summarize_comparison(
                    comparison_result
                )
                st.session_state["comparison_filings"] = {
                    "current": current_filing,
                    "previous": previous_filing,
                }
    except LookupError as error:
        # SEC_USER_AGENT 설정이 없는 경우.
        st.session_state["comparison_error"] = str(error)
    except (SecApiError, ValueError) as error:
        # sec_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
        st.session_state["comparison_error"] = str(error)
    except Exception:
        # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
        st.session_state["comparison_error"] = (
            "두 분기를 비교하는 중 예상하지 못한 문제가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

if st.session_state.get("comparison_error"):
    st.error(st.session_state["comparison_error"])
elif st.session_state.get("comparison_warning"):
    st.warning(st.session_state["comparison_warning"])
elif st.session_state.get("comparison") is not None:
    comparison = st.session_state["comparison"]
    summary = st.session_state["comparison_summary"] or {}
    comparison_filings = st.session_state.get("comparison_filings") or {}
    current_filing = comparison_filings.get("current") or {}
    previous_filing = comparison_filings.get("previous") or {}

    if comparison.empty:
        st.warning(
            "두 공시 모두에서 보유 종목을 찾지 못해 비교할 내용이 없습니다. "
            "잠시 후 다시 시도해 주세요."
        )
    else:
        # --- A. 비교 기준 ---------------------------------------------------
        st.markdown("**비교 기준**")
        current_column, previous_column = st.columns(2)
        current_column.markdown(
            f"현재 분기 (최신 공시)\n\n"
            f"- 제출일(filing_date): `{current_filing.get('filing_date', '-')}`\n"
            f"- 기준일(report_date): `{current_filing.get('report_date', '-')}`"
        )
        previous_column.markdown(
            f"이전 분기\n\n"
            f"- 제출일(filing_date): `{previous_filing.get('filing_date', '-')}`\n"
            f"- 기준일(report_date): `{previous_filing.get('report_date', '-')}`"
        )

        # --- B. 요약 지표 ---------------------------------------------------
        st.markdown("**요약 지표**")
        value_columns = st.columns(3)
        value_columns[0].metric(
            "현재 분기 전체 평가금액",
            f"{summary.get('current_total_value', 0):,.0f}",
        )
        value_columns[1].metric(
            "이전 분기 전체 평가금액",
            f"{summary.get('previous_total_value', 0):,.0f}",
        )
        value_columns[2].metric(
            "전체 평가금액 증감률",
            format_percent(summary.get("total_value_change_pct")),
        )
        st.caption(
            "평가금액은 **SEC Information Table의 reported value 필드를 사용합니다.** "
            "값은 환산하지 않고 공시 원문 그대로 표시합니다. "
            f"증감액은 {summary.get('total_value_change', 0):+,.0f} 입니다."
        )

        count_columns = st.columns(5)
        count_columns[0].metric(
            "신규 편입", f"{summary.get('new_position_count', 0)}개"
        )
        count_columns[1].metric(
            "보유 확대", f"{summary.get('increased_position_count', 0)}개"
        )
        count_columns[2].metric(
            "보유 축소", f"{summary.get('decreased_position_count', 0)}개"
        )
        count_columns[3].metric(
            "전량 매도", f"{summary.get('exited_position_count', 0)}개"
        )
        count_columns[4].metric(
            "유지", f"{summary.get('unchanged_position_count', 0)}개"
        )

        st.warning(
            "13F의 평가금액 변화는 **주가 변화와 보유수량 변화가 함께 반영된 값**입니다. "
            "따라서 실제 매수·매도 금액과 같지 않습니다. "
            "실제로 사고팔았는지는 보유수량(shares) 변화를 함께 확인해 주세요."
        )

        # --- C. 종목 변화 표 ------------------------------------------------
        st.markdown("**종목 변화 상세**")
        st.caption(
            "평가금액 관련 열은 SEC Information Table의 reported value 필드를 사용합니다. "
            "비중(previous_weight, current_weight)은 %, "
            "비중 변화(weight_change_pct_point)는 **%포인트**입니다. "
            "같은 회사라도 **일반 주식 보유와 Put/Call 옵션 보유, 수량 단위(share_type)가 "
            "다른 보유는 서로 다른 줄**로 비교합니다."
        )

        status_counts = comparison["change_status"].value_counts()
        tabs = st.tabs(
            [f"{status} ({int(status_counts.get(status, 0))})" for status in CHANGE_STATUS_ORDER]
        )

        for tab, status in zip(tabs, CHANGE_STATUS_ORDER):
            with tab:
                rows = comparison[comparison["change_status"] == status]
                if rows.empty:
                    st.info("해당 종목이 없습니다.")
                else:
                    st.dataframe(
                        rows[COMPARISON_DISPLAY_COLUMNS],
                        hide_index=True,
                        width="stretch",
                        height=table_height(len(rows)),
                        column_config=comparison_column_config(),
                    )

        # --- D. 변화 차트 ---------------------------------------------------
        st.markdown(f"**비중 변화가 큰 상위 {TOP_WEIGHT_CHANGE_COUNT}개 종목**")
        st.caption(
            "늘어난 종목과 줄어든 종목을 함께 보기 위해 비중 변화의 절대값이 큰 순서로 "
            "골랐습니다. 신규 편입과 전량 매도 종목도 포함됩니다. "
            "가로축 단위는 %포인트입니다."
        )

        chart_data = top_weight_changes(comparison)
        if chart_data.empty:
            st.info("비중이 변한 종목이 없습니다.")
        else:
            st.altair_chart(weight_change_chart(chart_data), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# AI 브리핑 (Gemini)
# ---------------------------------------------------------------------------

st.subheader("AI 포트폴리오 브리핑")

# 위에서 만든 두 분기 비교 결과가 있을 때만 이 기능을 씁니다.
briefing_comparison = st.session_state.get("comparison")
has_comparison = briefing_comparison is not None and not briefing_comparison.empty

if not has_comparison:
    st.info(
        "먼저 위의 '두 분기 변화 분석' 버튼을 눌러 비교 결과를 만든 뒤 "
        "AI 브리핑을 생성할 수 있습니다."
    )
else:
    st.caption(
        "위에서 Python이 계산한 두 분기 분석 결과(요약 지표와 주요 변화 종목)만 "
        "Gemini에 전달해 한국어 설명을 만듭니다. Gemini는 새로운 숫자를 계산하지 않습니다."
    )

    # 버튼을 눌렀을 때만 API를 호출합니다. 화면이 다시 그려져도 자동 호출되지 않습니다.
    if st.button("AI 브리핑 생성", type="primary"):
        st.session_state["ai_briefing"] = None
        st.session_state["ai_briefing_error"] = None

        try:
            # API 키와 모델명은 여기서 읽어 생성 함수에만 넘깁니다. 화면에 표시하지 않습니다.
            gemini_api_key, gemini_model = read_gemini_settings()

            briefing_filings = st.session_state.get("comparison_filings") or {}
            prompt = build_briefing_prompt(
                briefing_comparison,
                st.session_state.get("comparison_summary") or {},
                briefing_filings.get("current") or {},
                briefing_filings.get("previous") or {},
                manager["name"],
            )

            with st.spinner("Gemini가 분석 결과를 정리하는 중입니다..."):
                st.session_state["ai_briefing"] = generate_briefing(
                    prompt,
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                )
        except LookupError as error:
            # GEMINI_API_KEY 또는 GEMINI_MODEL 설정이 없는 경우.
            st.session_state["ai_briefing_error"] = str(error)
        except LlmApiError as error:
            # llm_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
            st.session_state["ai_briefing_error"] = str(error)
        except Exception:
            # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
            st.session_state["ai_briefing_error"] = (
                "AI 브리핑을 만드는 중 예상하지 못한 문제가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )

    if st.session_state.get("ai_briefing_error"):
        st.error(st.session_state["ai_briefing_error"])
    elif st.session_state.get("ai_briefing"):
        st.warning(
            "AI 생성 결과는 투자 권유가 아니며 제공된 13F 분석 데이터만 요약합니다. "
            "내용에 사실과 다른 부분이 있을 수 있으니 위의 표와 숫자를 함께 확인해 주세요."
        )
        render_ai_briefing_card(
            st.session_state["ai_briefing"], AI_BRIEFING_CARD_KEY
        )

st.divider()

# ---------------------------------------------------------------------------
# 기관 간 동일 분기 비교
#
# 위쪽 단일 기관 분석과는 완전히 별개의 화면입니다. 선택하는 기관도, 화면 상태 키도
# 나누어 두었기 때문에 여기서 기관을 바꿔도 위쪽 결과는 그대로 남습니다.
# ---------------------------------------------------------------------------

st.subheader("기관 간 동일 분기 비교")
st.caption(
    "두 기관이 동일한 기준일(report_date)에 보고한 13F 포트폴리오를 비교합니다."
)
st.caption(
    "기관마다 13F 제출 시점이 달라 제출일(filing_date)이 아닌 "
    f"기준일(report_date)로 짝을 맞춥니다. 공통 분기는 기관별 최근 공시 "
    f"{INSTITUTION_COMPARISON_FILING_LIMIT}건 범위에서 찾습니다."
)

institution_left_column, institution_right_column = st.columns(2)

left_institution_name = institution_left_column.selectbox(
    "기관 A",
    options=manager_names,
    index=institution_default_index(manager_names, DEFAULT_LEFT_INSTITUTION),
    key="institution_left_manager",
)
right_institution_name = institution_right_column.selectbox(
    "기관 B",
    options=manager_names,
    index=institution_default_index(
        manager_names, DEFAULT_RIGHT_INSTITUTION, avoid=DEFAULT_LEFT_INSTITUTION
    ),
    key="institution_right_manager",
)

# 두 기관 중 하나라도 바뀌면 이전 비교 결과만 지웁니다.
sync_institution_pair(
    st.session_state, left_institution_name, right_institution_name
)

# 같은 기관을 고르면 비교를 진행하지 않습니다(모든 종목이 공통 보유가 되어 의미가 없습니다).
left_institution = None
right_institution = None

if not can_compare_institutions(left_institution_name, right_institution_name):
    st.warning(
        "기관 A와 기관 B가 같습니다. 서로 다른 두 기관을 선택하면 비교할 수 있습니다."
    )
else:
    try:
        left_institution = find_manager(managers, left_institution_name)
        right_institution = find_manager(managers, right_institution_name)
    except (LookupError, KeyError) as error:
        st.error(str(error))

if left_institution is not None and right_institution is not None:
    name_left_column, name_right_column = st.columns(2)
    name_left_column.markdown(
        f"기관 A\n\n**{left_institution['name']}** (CIK {left_institution['cik']})"
    )
    name_right_column.markdown(
        f"기관 B\n\n**{right_institution['name']}** (CIK {right_institution['cik']})"
    )

    # --- 1단계: 두 기관이 함께 공시한 분기 찾기 ----------------------------
    # 버튼을 눌렀을 때만 SEC에 요청합니다. 화면이 다시 그려질 때 자동으로
    # 반복 조회하지 않게 하기 위함입니다.
    if st.button("공통 비교 분기 조회", key="institution_common_dates_button"):
        reset_institution_comparison_state(st.session_state)

        try:
            # User-Agent는 여기서 읽어 조회 함수에만 넘깁니다. 화면에 표시하지 않습니다.
            user_agent = read_sec_user_agent()

            with st.spinner("두 기관의 최근 13F 공시 목록을 불러오는 중입니다..."):
                # 위쪽 화면과 같은 캐시 함수를 씁니다. SEC 조회를 직접 하지 않습니다.
                left_filings = cached_recent_filings(
                    left_institution["cik"],
                    INSTITUTION_COMPARISON_FILING_LIMIT,
                    user_agent,
                )
                right_filings = cached_recent_filings(
                    right_institution["cik"],
                    INSTITUTION_COMPARISON_FILING_LIMIT,
                    user_agent,
                )

            st.session_state["institution_left_filings_by_date"] = (
                index_filings_by_report_date(left_filings)
            )
            st.session_state["institution_right_filings_by_date"] = (
                index_filings_by_report_date(right_filings)
            )
            st.session_state["institution_common_dates"] = find_common_report_dates(
                left_filings, right_filings
            )
            st.session_state["institution_comparison_left_name"] = left_institution[
                "name"
            ]
            st.session_state["institution_comparison_right_name"] = right_institution[
                "name"
            ]
        except LookupError as error:
            # SEC_USER_AGENT 설정이 없는 경우.
            st.session_state["institution_comparison_error"] = str(error)
        except (SecApiError, ValueError) as error:
            # sec_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
            st.session_state["institution_comparison_error"] = str(error)
        except Exception:
            # 예상하지 못한 오류. 내부 정보가 새지 않도록 상세 내용은 보여주지 않습니다.
            st.session_state["institution_comparison_error"] = (
                "공통 비교 분기를 찾는 중 예상하지 못한 문제가 발생했습니다. "
                "잠시 후 다시 시도해 주세요."
            )

    if st.session_state.get("institution_comparison_error"):
        st.error(st.session_state["institution_comparison_error"])

    institution_common_dates = st.session_state.get("institution_common_dates")

    if institution_common_dates is None:
        st.info(
            "위의 '공통 비교 분기 조회' 버튼을 눌러 두 기관이 함께 공시한 분기를 "
            "먼저 찾아 주세요."
        )
    elif not institution_common_dates:
        st.warning(
            f"{left_institution['name']}과 {right_institution['name']}이 함께 공시한 "
            f"기준일을 최근 {INSTITUTION_COMPARISON_FILING_LIMIT}건 범위에서 찾지 "
            "못했습니다. 다른 기관 조합으로 시도해 주세요."
        )
    else:
        # --- 2단계: 비교할 공통 기준일 고르기 -----------------------------
        # 목록은 find_common_report_dates가 최신순으로 돌려줍니다.
        selected_report_date = st.selectbox(
            "비교할 기준일(report_date)을 선택하세요",
            options=institution_common_dates,
            key="institution_report_date_choice",
        )

        # 기준일을 바꾸면 앞서 만든 AI 브리핑은 다른 분기의 설명이므로 지웁니다.
        # (비교 표와 요약 지표는 그대로 두고, 브리핑만 다시 만들 수 있게 합니다.)
        sync_institution_briefing_report_date(st.session_state, selected_report_date)

        left_comparison_filing = institution_filing_for_date(
            st.session_state.get("institution_left_filings_by_date"),
            selected_report_date,
        )
        right_comparison_filing = institution_filing_for_date(
            st.session_state.get("institution_right_filings_by_date"),
            selected_report_date,
        )

        filing_left_column, filing_right_column = st.columns(2)
        filing_left_column.markdown(
            format_institution_filing_summary(
                f"기관 A · {short_institution_name(left_institution['name'])}",
                left_comparison_filing,
            )
        )
        filing_right_column.markdown(
            format_institution_filing_summary(
                f"기관 B · {short_institution_name(right_institution['name'])}",
                right_comparison_filing,
            )
        )

        # --- 3단계: 보유 종목을 불러와 비교 실행 ---------------------------
        if st.button(
            "기관 포트폴리오 비교",
            type="primary",
            key="institution_compare_button",
        ):
            st.session_state["institution_comparison"] = None
            st.session_state["institution_comparison_summary"] = None
            st.session_state["institution_comparison_error"] = None
            st.session_state["institution_comparison_warning"] = None
            # 비교를 새로 실행하면, 앞서 만든 AI 브리핑은 옛 결과 기준이므로 지웁니다.
            reset_institution_ai_briefing_state(st.session_state)
            st.session_state["institution_selected_report_date"] = selected_report_date
            st.session_state["institution_comparison_filings"] = {
                "left": left_comparison_filing,
                "right": right_comparison_filing,
            }
            st.session_state["institution_comparison_left_name"] = left_institution[
                "name"
            ]
            st.session_state["institution_comparison_right_name"] = right_institution[
                "name"
            ]

            if not left_comparison_filing.get(
                "accession_number"
            ) or not right_comparison_filing.get("accession_number"):
                st.session_state["institution_comparison_warning"] = (
                    "선택한 기준일의 공시 정보를 찾지 못했습니다. "
                    "'공통 비교 분기 조회'를 다시 눌러 주세요."
                )
            else:
                try:
                    # User-Agent는 여기서 읽어 조회 함수에만 넘깁니다.
                    user_agent = read_sec_user_agent()

                    # 보유 종목도 위쪽 화면과 같은 캐시 함수를 씁니다.
                    with st.spinner("기관 A의 보유 종목을 불러오는 중입니다..."):
                        left_institution_holdings = cached_13f_holdings(
                            left_institution["cik"],
                            left_comparison_filing["accession_number"],
                            user_agent,
                        )

                    with st.spinner("기관 B의 보유 종목을 불러오는 중입니다..."):
                        right_institution_holdings = cached_13f_holdings(
                            right_institution["cik"],
                            right_comparison_filing["accession_number"],
                            user_agent,
                        )

                    with st.spinner("두 기관의 포트폴리오를 비교하는 중입니다..."):
                        institution_result = compare_institution_portfolios(
                            left_institution_holdings, right_institution_holdings
                        )
                        st.session_state["institution_comparison"] = institution_result
                        st.session_state["institution_comparison_summary"] = (
                            summarize_institution_comparison(institution_result)
                        )
                except LookupError as error:
                    # SEC_USER_AGENT 설정이 없는 경우.
                    st.session_state["institution_comparison_error"] = str(error)
                except (SecApiError, ValueError) as error:
                    # sec_client가 만들어 준 한국어 안내 메시지를 그대로 보여줍니다.
                    st.session_state["institution_comparison_error"] = str(error)
                except Exception:
                    # 예상하지 못한 오류. 상세 내용은 보여주지 않습니다.
                    st.session_state["institution_comparison_error"] = (
                        "두 기관을 비교하는 중 예상하지 못한 문제가 발생했습니다. "
                        "잠시 후 다시 시도해 주세요."
                    )

        # --- 4단계: 비교 결과 표시 ----------------------------------------
        if st.session_state.get("institution_comparison_warning"):
            st.warning(st.session_state["institution_comparison_warning"])
        elif st.session_state.get("institution_comparison") is not None:
            institution_comparison = st.session_state["institution_comparison"]
            institution_summary = (
                st.session_state.get("institution_comparison_summary") or {}
            )
            compared_left_name = (
                st.session_state.get("institution_comparison_left_name") or "기관 A"
            )
            compared_right_name = (
                st.session_state.get("institution_comparison_right_name") or "기관 B"
            )
            compared_report_date = (
                st.session_state.get("institution_selected_report_date") or "-"
            )

            if len(institution_comparison) == 0:
                st.warning(
                    "두 공시 모두에서 보유 종목을 찾지 못해 비교할 내용이 없습니다. "
                    "다른 기준일을 선택해 다시 시도해 주세요."
                )
            else:
                short_left = short_institution_name(compared_left_name)
                short_right = short_institution_name(compared_right_name)

                st.success(
                    f"기준일(report_date) `{compared_report_date}` 기준으로 "
                    f"**{compared_left_name}**(기관 A)와 "
                    f"**{compared_right_name}**(기관 B)를 비교했습니다."
                )

                # --- A. 요약 지표 -----------------------------------------
                st.markdown("**요약 지표**")
                institution_count_columns = st.columns(3)
                institution_count_columns[0].metric(
                    "공통 보유", f"{institution_summary.get('common_count', 0)}개"
                )
                institution_count_columns[1].metric(
                    f"{short_left} 단독",
                    f"{institution_summary.get('left_only_count', 0)}개",
                )
                institution_count_columns[2].metric(
                    f"{short_right} 단독",
                    f"{institution_summary.get('right_only_count', 0)}개",
                )

                institution_overlap_columns = st.columns(2)
                institution_overlap_columns[0].metric(
                    "종목 중복률",
                    format_overlap_percent(
                        institution_summary.get("security_overlap_pct")
                    ),
                    help="두 기관 포지션을 합친 개수 중 공통 보유가 차지하는 비율(%)",
                )
                institution_overlap_columns[1].metric(
                    "비중 기준 중복도",
                    format_overlap_percent(
                        institution_summary.get("weighted_overlap_pct")
                    ),
                    help=(
                        "공통 보유 종목마다 두 기관의 비중 중 작은 값을 더한 값(%). "
                        "포트폴리오의 몇 %가 겹치는지를 나타냅니다."
                    ),
                )
                st.caption(
                    "종목 중복률은 '몇 종목을 함께 들고 있는지', 비중 기준 중복도는 "
                    "'포트폴리오의 몇 %가 겹치는지'를 나타냅니다. 두 값은 서로 다른 "
                    "관점이므로 함께 확인해 주세요."
                )

                # --- B. 비교 결과 표 --------------------------------------
                st.markdown("**보유 종목 비교 상세**")
                st.caption(
                    "짝을 맞추는 기준은 CUSIP + 옵션 구분 + 수량 단위입니다. "
                    "같은 회사라도 일반 주식 보유와 Put/Call 옵션 보유는 서로 다른 "
                    "줄로 비교합니다. 한쪽만 보유한 종목의 반대쪽 값은 0으로 표시합니다. "
                    "평가금액은 SEC Information Table의 reported value 필드를 환산 없이 "
                    "그대로 표시하므로, 기관 간 비교는 비중(%)을 기준으로 봐 주세요."
                )

                institution_type_labels = {
                    HOLDING_TYPE_COMMON: "공통 보유",
                    HOLDING_TYPE_LEFT_ONLY: f"{short_left} 단독",
                    HOLDING_TYPE_RIGHT_ONLY: f"{short_right} 단독",
                }
                institution_type_rows = {
                    holding_type: institution_rows_by_type(
                        institution_comparison, holding_type
                    )
                    for holding_type in INSTITUTION_HOLDING_TYPE_ORDER
                }

                institution_tabs = st.tabs(
                    [
                        f"{institution_type_labels[holding_type]} "
                        f"({len(institution_type_rows[holding_type])})"
                        for holding_type in INSTITUTION_HOLDING_TYPE_ORDER
                    ]
                )

                for tab, holding_type in zip(
                    institution_tabs, INSTITUTION_HOLDING_TYPE_ORDER
                ):
                    with tab:
                        render_institution_holdings_table(
                            institution_type_rows[holding_type]
                        )

                # --- C. 해석 시 주의 --------------------------------------
                st.warning(
                    "13F는 **분기 말(기준일) 기준 보유 현황**이며 실시간 포트폴리오가 "
                    "아닙니다. 공시 이후의 매매는 반영되어 있지 않습니다."
                )
                st.caption(
                    "비교 결과는 13F 공시 대상 증권(주로 미국 상장 주식과 일부 옵션 등) "
                    "범위에 한정됩니다. 채권·현물·해외 주식·공매도 포지션 등 13F에 "
                    "보고되지 않는 자산은 포함되지 않습니다."
                )
                st.caption(
                    "공통 보유는 같은 시점에 같은 종목을 보고했다는 사실만 알려 주며, "
                    "두 기관이 동일한 투자 의도나 전략을 가졌다는 뜻은 아닙니다. "
                    "보유 이유·기간·연계 포지션은 공시 데이터로 알 수 없습니다."
                )

                # --- D. 기관 비교 AI 브리핑 --------------------------------
                # 위 비교 결과가 있을 때만(이 else 블록 안) 보여 주므로, 비교
                # 결과가 없으면 브리핑을 만들 수 없습니다.
                st.markdown("**AI 기관 비교 브리핑**")
                st.caption(INSTITUTION_BRIEFING_NOTICE)
                st.caption(
                    "Gemini에는 위에서 Python이 계산한 요약 지표와 상위 종목 목록만 "
                    "전달합니다. 공시 원문(XML)이나 전체 비교 표는 전달하지 않으며, "
                    "Gemini는 새로운 숫자를 계산하지 않습니다."
                )

                # 버튼을 눌렀을 때만 API를 호출합니다. 화면이 다시 그려져도
                # 자동으로 다시 호출되지 않습니다.
                if st.button(
                    "AI 기관 비교 브리핑 생성",
                    type="primary",
                    key="institution_ai_briefing_button",
                ):
                    run_institution_ai_briefing(st.session_state)

                if st.session_state.get("institution_ai_briefing_error"):
                    st.error(st.session_state["institution_ai_briefing_error"])
                elif st.session_state.get("institution_ai_briefing"):
                    st.warning(
                        "AI 생성 결과는 투자 권유가 아니며 제공된 기관 비교 데이터만 "
                        "설명합니다. 내용에 사실과 다른 부분이 있을 수 있으니 위의 "
                        "표와 숫자를 함께 확인해 주세요."
                    )
                    render_ai_briefing_card(
                        st.session_state["institution_ai_briefing"],
                        INSTITUTION_AI_BRIEFING_CARD_KEY,
                    )

st.divider()

st.info(
    "본 프로젝트는 교육용 PoC(개념 검증)입니다. "
    "투자 자문이나 투자 권유가 아니며, 분석 결과의 정확성을 보장하지 않습니다."
)
