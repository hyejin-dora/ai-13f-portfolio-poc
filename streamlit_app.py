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

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from services.llm_client import (
    LlmApiError,
    build_briefing_prompt,
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

# 비중 상위 몇 개 종목을 따로 보여줄지.
TOP_HOLDINGS_COUNT = 10

# SEC 조회 결과를 얼마나 오래 재사용할지(초). 6시간.
# 13F는 분기마다 나오는 공시라 하루 안에 내용이 바뀌는 일이 거의 없으므로,
# 같은 값을 반복해서 내려받지 않도록 넉넉하게 잡았습니다.
SEC_CACHE_TTL_SECONDS = 6 * 60 * 60

# 행이 많은 표(예: 보유 종목이 수백 개인 기관)의 표시 높이(픽셀).
# 높이를 정해 두면 표 안에서만 스크롤되어 화면이 지나치게 길어지지 않습니다.
LARGE_TABLE_HEIGHT = 520

# 몇 행을 넘을 때부터 위 높이를 적용할지. 이보다 적으면 높이를 정하지 않아
# (자동 높이) 표 아래에 빈 공간이 생기지 않습니다.
LARGE_TABLE_ROW_THRESHOLD = 15

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


def table_height(row_count: int) -> int | None:
    """표에 지정할 높이를 정합니다.

    보유 종목이 많은 기관(예: Bridgewater)은 표가 화면을 끝없이 늘리지 않도록
    높이를 정해 표 안에서 스크롤되게 합니다. 행이 적으면 None을 돌려주어
    Streamlit 기본(자동 높이)을 그대로 쓰게 합니다. 어느 쪽이든 행을 잘라내지
    않으므로 전체 데이터는 그대로 유지됩니다.
    """
    if row_count > LARGE_TABLE_ROW_THRESHOLD:
        return LARGE_TABLE_HEIGHT
    return None


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

    return (bars + zero_line).properties(height=alt.Step(26))


# ---------------------------------------------------------------------------
# 화면 구성
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI 13F 포트폴리오 분석 PoC", page_icon="📊")

st.title("📊 AI 13F 포트폴리오 분석 PoC")

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
        st.markdown(st.session_state["ai_briefing"])

st.divider()

st.info(
    "본 프로젝트는 교육용 PoC(개념 검증)입니다. "
    "투자 자문이나 투자 권유가 아니며, 분석 결과의 정확성을 보장하지 않습니다."
)
