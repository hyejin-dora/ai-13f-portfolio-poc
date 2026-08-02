"""AI 13F 포트폴리오 분석 PoC - 메인 화면.

이번 단계에서는 SEC EDGAR에서 최근 13F-HR 공시 목록을 조회해 표로 보여줍니다.
보유 종목 상세 분석과 Gemini API 설명 생성은 다음 단계에서 추가합니다.

화면(이 파일)은 사용자에게 보여주는 일만 하고,
실제 데이터 수집은 services/sec_client.py가 담당합니다.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from services.sec_client import SecApiError, get_recent_13f_filings

# 이번 단계의 분석 대상 운용사. data/managers.csv에서 이 이름으로 찾습니다.
TARGET_MANAGER = "Berkshire Hathaway"

# 조회할 공시 건수.
FILING_LIMIT = 2

MANAGERS_CSV = Path(__file__).parent / "data" / "managers.csv"

# 화면에 표시할 표의 열 순서와 한국어 제목.
COLUMN_LABELS = {
    "accession_number": "접수번호 (accession_number)",
    "filing_date": "제출일 (filing_date)",
    "report_date": "기준일 (report_date)",
    "primary_document": "주요 문서 (primary_document)",
}


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


def to_display_table(filings: list[dict]) -> pd.DataFrame:
    """조회 결과를 화면에 표시할 표 형태로 정리합니다."""
    table = pd.DataFrame(filings, columns=list(COLUMN_LABELS))
    return table.rename(columns=COLUMN_LABELS)


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

# 운용사 정보 읽기. 파일이 없거나 대상이 없으면 여기서 안내하고 멈춥니다.
try:
    manager = find_manager(load_managers(), TARGET_MANAGER)
except FileNotFoundError:
    st.error(
        "운용사 목록 파일(data/managers.csv)을 찾을 수 없습니다. "
        "파일이 있는지 확인해 주세요."
    )
    st.stop()
except (LookupError, KeyError, pd.errors.ParserError):
    st.error(
        "운용사 목록 파일(data/managers.csv)을 읽을 수 없습니다. "
        f"'{TARGET_MANAGER}' 항목과 name, cik 열이 있는지 확인해 주세요."
    )
    st.stop()

st.write(f"{manager['name']} (CIK {manager['cik']})")

st.divider()

st.subheader(f"최근 13F 공시 {FILING_LIMIT}건")
st.caption(
    "13F-HR은 운용 규모가 큰 기관투자자가 분기마다 제출하는 보유 종목 보고서입니다. "
    "정정 공시(13F-HR/A)는 제외하고 정기 공시만 조회합니다."
)

if st.button("최근 13F 공시 조회", type="primary"):
    # 조회 결과와 오류 메시지를 화면 상태에 보관합니다.
    # 다른 버튼을 눌러 화면이 다시 그려져도 표가 사라지지 않게 하기 위함입니다.
    st.session_state["filings"] = None
    st.session_state["error"] = None

    try:
        # User-Agent는 여기서 읽어 조회 함수에 바로 넘깁니다. 화면에 표시하지 않습니다.
        user_agent = read_sec_user_agent()

        with st.spinner("SEC EDGAR에서 공시 목록을 불러오는 중입니다..."):
            st.session_state["filings"] = get_recent_13f_filings(
                manager["cik"],
                user_agent=user_agent,
                limit=FILING_LIMIT,
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

st.info(
    "본 프로젝트는 교육용 PoC(개념 검증)입니다. "
    "투자 자문이나 투자 권유가 아니며, 분석 결과의 정확성을 보장하지 않습니다."
)
