"""streamlit_app.py의 기관투자자 선택 및 SEC 조회 캐시 테스트.

실제 화면을 띄우지 않고, 운용사 목록 읽기와 선택 상태 처리,
그리고 SEC 조회 캐시 래퍼의 동작만 확인합니다.
Streamlit 위젯은 `streamlit run` 없이 불러오면 기본값을 돌려주므로,
이 파일은 모듈을 그대로 import 해서 안에 있는 함수만 검증합니다.

SEC에 실제로 접속하지 않습니다. 조회 함수는 모두 가짜(mock)로 바꿔치기합니다.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import streamlit_app
from services.institution_comparison import (
    HOLDING_TYPE_COMMON,
    HOLDING_TYPE_LEFT_ONLY,
    HOLDING_TYPE_RIGHT_ONLY,
    compare_institution_portfolios,
    find_common_report_dates,
    index_filings_by_report_date,
    summarize_institution_comparison,
)
from services.llm_client import LlmApiError
from streamlit_app import (
    ANALYSIS_STATE_KEYS,
    DEFAULT_MANAGER,
    INSTITUTION_AI_BRIEFING_STATE_KEYS,
    INSTITUTION_COMPARISON_STATE_KEYS,
    INSTITUTION_DISPLAY_COLUMNS,
    cached_13f_holdings,
    cached_recent_filings,
    can_compare_institutions,
    clear_sec_caches,
    default_manager_index,
    find_manager,
    format_overlap_percent,
    hero_background_css,
    hero_html,
    hero_image_data_uri,
    institution_briefing_payload_from_state,
    institution_column_config,
    institution_default_index,
    institution_filing_for_date,
    institution_rows_by_type,
    load_managers,
    manager_options,
    render_institution_holdings_table,
    reset_analysis_state,
    reset_institution_ai_briefing_state,
    reset_institution_comparison_state,
    run_institution_ai_briefing,
    short_institution_name,
    sync_institution_briefing_report_date,
    sync_institution_pair,
    sync_selected_manager,
    table_height,
    theme_variables_css,
)

# data/managers.csv에 반드시 들어 있어야 하는 기관과 CIK.
EXPECTED_MANAGERS = {
    "Berkshire Hathaway": "0001067983",
    "Pershing Square Capital Management": "0001336528",
    "Bridgewater Associates": "0001350694",
}

# 기관을 바꿀 때 반드시 초기화되어야 하는 화면 상태 키.
EXPECTED_RESET_KEYS = [
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
]


@pytest.fixture
def managers() -> pd.DataFrame:
    """실제 data/managers.csv를 읽어 온 운용사 목록."""
    # @st.cache_data가 이전 테스트의 결과를 재사용하지 않도록 캐시를 비웁니다.
    load_managers.clear()
    return load_managers()


# ---------------------------------------------------------------------------
# 운용사 목록 읽기
# ---------------------------------------------------------------------------


def test_managers_csv_has_name_and_cik_columns(managers):
    """열 구조는 기존과 같은 name, cik 두 개를 유지해야 한다."""
    assert list(managers.columns) == ["name", "cik"]


def test_load_managers_includes_three_managers(managers):
    """세 기관이 모두 목록에 들어 있어야 한다."""
    names = managers["name"].tolist()

    for expected_name in EXPECTED_MANAGERS:
        assert expected_name in names


def test_load_managers_keeps_leading_zero_in_cik(managers):
    """CIK는 문자열로 읽어 앞자리 0이 사라지지 않아야 한다."""
    for cik in managers["cik"]:
        assert isinstance(cik, str)
        # 13F 제출 기관의 CIK는 앞자리 0이 채워진 10자리 문자열입니다.
        assert len(cik) == 10
        assert cik.startswith("0")


def test_load_managers_cik_values_are_exact(managers):
    """각 기관의 CIK가 공시 기준 값과 정확히 같아야 한다."""
    loaded = dict(zip(managers["name"], managers["cik"]))

    assert loaded == EXPECTED_MANAGERS


# ---------------------------------------------------------------------------
# 이름으로 기관 찾기
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "cik"), sorted(EXPECTED_MANAGERS.items()))
def test_find_manager_returns_matching_cik(managers, name, cik):
    """기관명을 주면 그 기관의 name과 CIK를 돌려주어야 한다."""
    found = find_manager(managers, name)

    assert found == {"name": name, "cik": cik}


def test_find_manager_raises_for_unknown_name(managers):
    """목록에 없는 기관명을 주면 기존과 같이 LookupError를 내야 한다."""
    with pytest.raises(LookupError):
        find_manager(managers, "존재하지 않는 운용사")


def test_find_manager_raises_for_empty_name(managers):
    """빈 이름도 찾을 수 없는 이름으로 처리되어야 한다."""
    with pytest.raises(LookupError):
        find_manager(managers, "")


# ---------------------------------------------------------------------------
# 선택 상자에 넘길 목록과 기본값
# ---------------------------------------------------------------------------


def test_manager_options_returns_names_in_file_order(managers):
    """선택 상자 목록은 CSV에 적힌 순서를 그대로 따라야 한다."""
    assert manager_options(managers) == managers["name"].tolist()


def test_manager_options_raises_when_list_is_empty():
    """운용사가 한 곳도 없으면 고를 대상이 없으므로 LookupError를 내야 한다."""
    empty = pd.DataFrame({"name": [], "cik": []})

    with pytest.raises(LookupError):
        manager_options(empty)


def test_manager_options_raises_when_name_column_is_missing():
    """name 열이 없는 파일은 읽을 수 없다고 알려야 한다."""
    wrong_columns = pd.DataFrame({"manager": ["A"], "cik": ["0000000001"]})

    with pytest.raises(KeyError):
        manager_options(wrong_columns)


def test_default_manager_index_points_to_berkshire(managers):
    """기본 선택값은 Berkshire Hathaway여야 한다."""
    names = manager_options(managers)

    assert names[default_manager_index(names)] == DEFAULT_MANAGER
    assert DEFAULT_MANAGER == "Berkshire Hathaway"


def test_default_manager_index_falls_back_to_first_item():
    """기본 기관이 목록에 없으면 첫 번째 기관을 고른다."""
    names = ["Bridgewater Associates", "Pershing Square Capital Management"]

    assert default_manager_index(names) == 0


# ---------------------------------------------------------------------------
# 기관 변경 시 상태 초기화
# ---------------------------------------------------------------------------


def filled_state() -> dict:
    """분석을 한 번 마친 뒤의 화면 상태 예시."""
    state = {key: f"{key}-이전 결과" for key in EXPECTED_RESET_KEYS}
    state["active_manager_name"] = "Berkshire Hathaway"
    # 초기화 대상이 아닌 값(공시 선택 위치)도 함께 넣어 둡니다.
    state["selected_filing_index"] = 1
    return state


def test_analysis_state_keys_cover_every_required_key():
    """초기화 대상 목록에 필요한 상태 키가 모두 들어 있어야 한다."""
    assert set(ANALYSIS_STATE_KEYS) == set(EXPECTED_RESET_KEYS)


def test_reset_analysis_state_clears_all_analysis_keys():
    """reset_analysis_state는 분석 결과 키를 모두 비워야 한다."""
    state = filled_state()

    reset_analysis_state(state)

    for key in EXPECTED_RESET_KEYS:
        assert state[key] is None


def test_changing_manager_resets_previous_results():
    """기관 A에서 기관 B로 바꾸면 이전 기관의 결과가 남지 않아야 한다."""
    state = filled_state()

    changed = sync_selected_manager(state, "Bridgewater Associates")

    assert changed is True
    for key in EXPECTED_RESET_KEYS:
        assert state[key] is None
    assert state["active_manager_name"] == "Bridgewater Associates"


def test_rerun_with_same_manager_keeps_results():
    """같은 기관에서 화면이 다시 실행될 때는 결과가 사라지지 않아야 한다."""
    state = filled_state()
    before = dict(state)

    changed = sync_selected_manager(state, "Berkshire Hathaway")

    assert changed is False
    assert state == before


def test_first_run_records_selected_manager():
    """처음 화면을 열면 고른 기관 이름을 기억해 두어야 한다."""
    state = {}

    changed = sync_selected_manager(state, DEFAULT_MANAGER)

    assert changed is True
    assert state["active_manager_name"] == DEFAULT_MANAGER


def test_changing_manager_keeps_unrelated_state():
    """초기화 대상이 아닌 값은 그대로 두어야 한다."""
    state = filled_state()

    sync_selected_manager(state, "Pershing Square Capital Management")

    assert state["selected_filing_index"] == 1


def test_switching_back_and_forth_resets_each_time():
    """A → B → A로 바꿀 때마다 매번 초기화되어야 한다."""
    state = filled_state()

    assert sync_selected_manager(state, "Bridgewater Associates") is True

    state["filings"] = "Bridgewater 조회 결과"
    assert sync_selected_manager(state, "Berkshire Hathaway") is True
    assert state["filings"] is None


# ---------------------------------------------------------------------------
# 기존 기능 유지 확인
# ---------------------------------------------------------------------------


def test_target_manager_constant_is_removed():
    """기관명이 하드코딩된 TARGET_MANAGER는 더 이상 없어야 한다."""
    assert not hasattr(streamlit_app, "TARGET_MANAGER")


def test_filing_limit_is_unchanged():
    """공시 조회 건수 등 기존 설정은 이번 단계에서 바뀌지 않아야 한다."""
    assert streamlit_app.FILING_LIMIT == 2
    assert streamlit_app.TOP_HOLDINGS_COUNT == 10


# ---------------------------------------------------------------------------
# SEC 조회 캐시
# ---------------------------------------------------------------------------

# 캐시 테스트에 쓰는 예시 값. 실제 SEC 응답과 같은 구조만 갖추면 됩니다.
SAMPLE_CIK = "0001067983"
SAMPLE_USER_AGENT = "Test Runner test@example.com"
SAMPLE_ACCESSION = "0000950123-25-008888"

SAMPLE_FILINGS = [
    {
        "accession_number": SAMPLE_ACCESSION,
        "filing_date": "2025-02-14",
        "report_date": "2024-12-31",
        "primary_document": "primary_doc.xml",
    },
    {
        "accession_number": "0000950123-24-011111",
        "filing_date": "2024-11-14",
        "report_date": "2024-09-30",
        "primary_document": "primary_doc.xml",
    },
]

SAMPLE_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 75000000,
        "shares": 300000000,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 24000000,
        "shares": 400000000,
        "share_type": "SH",
        "put_call": "",
    },
]


@pytest.fixture(autouse=True)
def empty_sec_caches():
    """각 테스트가 빈 캐시에서 시작하고, 끝난 뒤에도 캐시를 남기지 않게 합니다."""
    clear_sec_caches()
    yield
    clear_sec_caches()


def test_cache_ttl_is_six_hours():
    """캐시 유효시간은 6시간이어야 한다."""
    assert streamlit_app.SEC_CACHE_TTL_SECONDS == 6 * 60 * 60


def test_cached_recent_filings_passes_arguments_to_sec_client():
    """공시 목록 캐시는 cik, limit, user_agent를 SEC 함수에 그대로 넘겨야 한다."""
    with patch("streamlit_app.get_recent_13f_filings") as mock_get:
        mock_get.return_value = SAMPLE_FILINGS

        result = cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)

    mock_get.assert_called_once_with(SAMPLE_CIK, user_agent=SAMPLE_USER_AGENT, limit=2)
    # 반환값 구조는 기존 함수와 같은 list[dict]를 유지해야 합니다.
    assert isinstance(result, list)
    assert result == SAMPLE_FILINGS


def test_cached_recent_filings_reuses_result_for_same_arguments():
    """같은 인자로 다시 부르면 SEC 함수를 또 호출하지 않아야 한다."""
    with patch("streamlit_app.get_recent_13f_filings") as mock_get:
        mock_get.return_value = SAMPLE_FILINGS

        first = cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)
        second = cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)

    assert mock_get.call_count == 1
    assert first == second == SAMPLE_FILINGS


@pytest.mark.parametrize(
    "other_arguments",
    [
        ("0001350694", 2, SAMPLE_USER_AGENT),  # 다른 기관
        (SAMPLE_CIK, 4, SAMPLE_USER_AGENT),  # 다른 조회 건수
        (SAMPLE_CIK, 2, "Other Runner other@example.com"),  # 다른 User-Agent
    ],
)
def test_cached_recent_filings_refetches_when_any_key_differs(other_arguments):
    """cik, limit, user_agent 중 하나만 달라도 SEC에 새로 요청해야 한다."""
    with patch("streamlit_app.get_recent_13f_filings") as mock_get:
        mock_get.return_value = SAMPLE_FILINGS

        cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)
        cached_recent_filings(*other_arguments)

    assert mock_get.call_count == 2


def test_cached_13f_holdings_passes_arguments_to_sec_client():
    """보유 종목 캐시는 cik, accession_number, user_agent를 그대로 넘겨야 한다."""
    with patch("streamlit_app.get_13f_holdings") as mock_get:
        mock_get.return_value = SAMPLE_HOLDINGS

        result = cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)

    mock_get.assert_called_once_with(
        SAMPLE_CIK, SAMPLE_ACCESSION, user_agent=SAMPLE_USER_AGENT
    )
    # 반환값은 기존 holdings 목록 구조를 그대로 유지해야 합니다.
    assert isinstance(result, list)
    assert result == SAMPLE_HOLDINGS


def test_cached_13f_holdings_does_not_refetch_same_filing():
    """같은 CIK·accession_number를 다시 요청하면 SEC XML을 재호출하지 않아야 한다."""
    with patch("streamlit_app.get_13f_holdings") as mock_get:
        mock_get.return_value = SAMPLE_HOLDINGS

        first = cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)
        second = cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)

    assert mock_get.call_count == 1
    assert first == second == SAMPLE_HOLDINGS


def test_cached_13f_holdings_refetches_for_other_accession_number():
    """다른 공시(분기)를 요청하면 그 공시는 새로 받아와야 한다."""
    with patch("streamlit_app.get_13f_holdings") as mock_get:
        mock_get.return_value = SAMPLE_HOLDINGS

        cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)
        cached_13f_holdings(SAMPLE_CIK, "0000950123-24-011111", SAMPLE_USER_AGENT)

    assert mock_get.call_count == 2


def test_two_quarter_comparison_reuses_holdings_already_fetched():
    """단일 공시 조회와 두 분기 비교가 같은 캐시를 쓰는지 확인한다.

    '보유 종목 조회'로 최신 공시를 본 뒤 '두 분기 변화 분석'을 하면,
    최신 공시는 캐시에 있으므로 이전 분기 공시만 새로 받아와야 합니다.
    """
    current = SAMPLE_FILINGS[0]["accession_number"]
    previous = SAMPLE_FILINGS[1]["accession_number"]

    with patch("streamlit_app.get_13f_holdings") as mock_get:
        mock_get.return_value = SAMPLE_HOLDINGS

        # 1) 단일 공시 조회
        cached_13f_holdings(SAMPLE_CIK, current, SAMPLE_USER_AGENT)
        # 2) 두 분기 비교
        cached_13f_holdings(SAMPLE_CIK, current, SAMPLE_USER_AGENT)
        cached_13f_holdings(SAMPLE_CIK, previous, SAMPLE_USER_AGENT)

    # 공시 3번 요청했지만 서로 다른 공시는 2건이므로 SEC 호출도 2번이어야 합니다.
    assert mock_get.call_count == 2
    requested = [call.args[1] for call in mock_get.call_args_list]
    assert requested == [current, previous]


# ---------------------------------------------------------------------------
# 캐시 초기화
# ---------------------------------------------------------------------------


def test_clear_sec_caches_forces_new_filings_request():
    """캐시를 비우면 공시 목록을 SEC에서 다시 받아와야 한다."""
    with patch("streamlit_app.get_recent_13f_filings") as mock_get:
        mock_get.return_value = SAMPLE_FILINGS

        cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)
        clear_sec_caches()
        cached_recent_filings(SAMPLE_CIK, 2, SAMPLE_USER_AGENT)

    assert mock_get.call_count == 2


def test_clear_sec_caches_forces_new_holdings_request():
    """캐시를 비우면 보유 종목도 SEC에서 다시 받아와야 한다."""
    with patch("streamlit_app.get_13f_holdings") as mock_get:
        mock_get.return_value = SAMPLE_HOLDINGS

        cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)
        clear_sec_caches()
        cached_13f_holdings(SAMPLE_CIK, SAMPLE_ACCESSION, SAMPLE_USER_AGENT)

    assert mock_get.call_count == 2


def test_clear_sec_caches_keeps_manager_list_cache(monkeypatch):
    """캐시 초기화는 SEC 조회 캐시만 비우고 운용사 목록 캐시는 남겨야 한다.

    앱 전체 캐시(st.cache_data.clear())를 지우지 않는다는 것을 확인하기 위해,
    운용사 목록을 한 번 읽어 캐시에 담은 뒤 파일 읽기를 실패하게 만들어 둡니다.
    캐시가 남아 있다면 파일을 다시 읽지 않으므로 오류 없이 값을 돌려줍니다.
    """
    load_managers.clear()
    before = load_managers()

    def fail_to_read(*args, **kwargs):
        raise AssertionError("운용사 목록 캐시가 지워져 파일을 다시 읽으려 했습니다.")

    monkeypatch.setattr(pd, "read_csv", fail_to_read)

    clear_sec_caches()

    assert load_managers().equals(before)


# ---------------------------------------------------------------------------
# 행이 많은 표 표시
# ---------------------------------------------------------------------------


def test_table_height_is_limited_for_many_rows():
    """보유 종목이 많으면(예: Bridgewater) 표 높이를 정해 스크롤되게 한다."""
    assert table_height(1000) == streamlit_app.LARGE_TABLE_HEIGHT
    assert table_height(streamlit_app.LARGE_TABLE_ROW_THRESHOLD + 1) == (
        streamlit_app.LARGE_TABLE_HEIGHT
    )


def test_table_height_is_automatic_for_few_rows():
    """행이 적으면 내용에 맞춘 높이를 써서 표 아래에 빈 공간이 생기지 않는다."""
    assert table_height(0) == streamlit_app.AUTO_TABLE_HEIGHT
    assert table_height(streamlit_app.LARGE_TABLE_ROW_THRESHOLD) == (
        streamlit_app.AUTO_TABLE_HEIGHT
    )


def _assert_valid_streamlit_height(value):
    """Streamlit이 표 높이로 허용하는 값(양의 정수, "content", "stretch")인지 확인한다."""
    assert value is not None
    if isinstance(value, str):
        assert value in {"content", "stretch"}
    else:
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value > 0


def test_table_height_never_returns_none_for_empty_table():
    """0행(조회 결과가 없는 표)이어도 None이 아니라 유효한 높이를 준다."""
    _assert_valid_streamlit_height(table_height(0))


def test_table_height_never_returns_none_for_single_row():
    """1행짜리 표도 None이 아니라 유효한 높이를 준다."""
    _assert_valid_streamlit_height(table_height(1))


def test_table_height_is_valid_for_small_tables():
    """행이 적은 구간 전체에서 유효한 높이를 준다."""
    for row_count in range(0, streamlit_app.LARGE_TABLE_ROW_THRESHOLD + 1):
        _assert_valid_streamlit_height(table_height(row_count))


def test_table_height_is_positive_integer_for_large_tables():
    """행이 많으면 양의 정수(픽셀) 높이를 주어 표 안에서 스크롤되게 한다."""
    for row_count in (
        streamlit_app.LARGE_TABLE_ROW_THRESHOLD + 1,
        200,
        5000,
    ):
        height = table_height(row_count)
        _assert_valid_streamlit_height(height)
        assert isinstance(height, int)
        assert height > 0


def test_cache_cleared_message_tells_user_to_search_again():
    """캐시 초기화 뒤에는 다시 조회해야 한다는 안내가 있어야 한다."""
    message = streamlit_app.CACHE_CLEARED_MESSAGE

    assert "캐시" in message
    assert "다시" in message


def test_cache_reset_does_not_touch_analysis_state():
    """캐시 초기화 대상에 화면 상태 키가 섞여 들어가지 않아야 한다.

    기관 선택이나 이미 화면에 나온 분석 결과는 캐시 초기화로 지우지 않습니다.
    """
    state = filled_state()

    clear_sec_caches()

    assert state["active_manager_name"] == "Berkshire Hathaway"
    for key in EXPECTED_RESET_KEYS:
        assert state[key] is not None


# ---------------------------------------------------------------------------
# 기관 간 동일 분기 비교 화면
#
# SEC에 접속하지 않습니다. 공시 목록과 보유 종목은 아래 예시 값을 씁니다.
# ---------------------------------------------------------------------------

# 기관 A(왼쪽)의 최근 공시 예시. 최신순으로 내려온다고 가정합니다.
LEFT_FILINGS = [
    {
        "accession_number": "0000950123-25-000001",
        "filing_date": "2025-05-15",
        "report_date": "2025-03-31",
        "primary_document": "primary_doc.xml",
    },
    {
        "accession_number": "0000950123-25-000002",
        "filing_date": "2025-02-14",
        "report_date": "2024-12-31",
        "primary_document": "primary_doc.xml",
    },
]

# 기관 B(오른쪽)의 최근 공시 예시. 제출일은 다르지만 기준일 하나가 겹칩니다.
RIGHT_FILINGS = [
    {
        "accession_number": "0000950123-25-000101",
        "filing_date": "2025-05-12",
        "report_date": "2025-03-31",
        "primary_document": "primary_doc.xml",
    },
    {
        "accession_number": "0000950123-24-000102",
        "filing_date": "2024-08-14",
        "report_date": "2024-06-30",
        "primary_document": "primary_doc.xml",
    },
]

LEFT_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 60000,
        "shares": 600,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 40000,
        "shares": 400,
        "share_type": "SH",
        "put_call": "",
    },
]

RIGHT_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 30000,
        "shares": 300,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "CHIPOTLE MEXICAN GRILL INC",
        "class_title": "COM",
        "cusip": "169656105",
        "reported_value": 70000,
        "shares": 100,
        "share_type": "SH",
        "put_call": "",
    },
]

# 기관 비교 화면이 만들어 두는 상태 키(초기화 대상).
EXPECTED_INSTITUTION_RESET_KEYS = [
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
    "institution_ai_briefing",
    "institution_ai_briefing_error",
]

# 그중 기관 비교 AI 브리핑만 담는 키.
EXPECTED_INSTITUTION_BRIEFING_KEYS = [
    "institution_ai_briefing",
    "institution_ai_briefing_error",
]


def sample_institution_comparison() -> pd.DataFrame:
    """예시 보유 종목으로 만든 두 기관 비교 결과."""
    return compare_institution_portfolios(LEFT_HOLDINGS, RIGHT_HOLDINGS)


def filled_institution_state() -> dict:
    """단일 기관 분석과 기관 비교를 모두 한 번씩 마친 뒤의 화면 상태 예시."""
    state = filled_state()
    for key in EXPECTED_INSTITUTION_RESET_KEYS:
        state[key] = f"{key}-이전 비교 결과"
    state["active_institution_pair"] = (
        "Berkshire Hathaway",
        "Pershing Square Capital Management",
    )
    return state


# --- 조회 범위 설정 --------------------------------------------------------


def test_institution_comparison_filing_limit_is_eight():
    """기관 비교 화면은 최근 8건 범위에서 공통 분기를 찾아야 한다."""
    assert streamlit_app.INSTITUTION_COMPARISON_FILING_LIMIT == 8


def test_single_manager_filing_limit_is_not_affected():
    """단일 기관 분석의 조회 건수는 그대로 2건이어야 한다."""
    assert streamlit_app.FILING_LIMIT == 2
    assert (
        streamlit_app.INSTITUTION_COMPARISON_FILING_LIMIT != streamlit_app.FILING_LIMIT
    )


def test_institution_default_pair_is_berkshire_and_pershing():
    """기본 비교 대상은 Berkshire(기관 A)와 Pershing Square(기관 B)여야 한다."""
    assert streamlit_app.DEFAULT_LEFT_INSTITUTION == "Berkshire Hathaway"
    assert (
        streamlit_app.DEFAULT_RIGHT_INSTITUTION
        == "Pershing Square Capital Management"
    )


def test_institution_default_index_points_to_each_default(managers):
    """두 선택 상자의 기본값이 각각 지정한 기관을 가리켜야 한다."""
    names = manager_options(managers)

    left_index = institution_default_index(
        names, streamlit_app.DEFAULT_LEFT_INSTITUTION
    )
    right_index = institution_default_index(
        names,
        streamlit_app.DEFAULT_RIGHT_INSTITUTION,
        avoid=streamlit_app.DEFAULT_LEFT_INSTITUTION,
    )

    assert names[left_index] == "Berkshire Hathaway"
    assert names[right_index] == "Pershing Square Capital Management"
    # 처음 화면을 열었을 때 두 상자가 같은 기관을 가리키지 않아야 합니다.
    assert left_index != right_index


def test_institution_default_index_avoids_the_other_default():
    """기본 기관이 목록에 없으면 반대쪽 기본 기관과 다른 기관을 골라야 한다."""
    names = ["Berkshire Hathaway", "Bridgewater Associates"]

    index = institution_default_index(
        names, "Pershing Square Capital Management", avoid="Berkshire Hathaway"
    )

    assert names[index] == "Bridgewater Associates"


# --- 기관 비교 전용 상태 초기화 --------------------------------------------


def test_institution_state_keys_cover_every_required_key():
    """초기화 대상 목록에 기관 비교 상태 키가 모두 들어 있어야 한다."""
    assert set(INSTITUTION_COMPARISON_STATE_KEYS) == set(
        EXPECTED_INSTITUTION_RESET_KEYS
    )


def test_institution_state_keys_do_not_overlap_analysis_keys():
    """단일 기관 분석 상태 키와 이름이 겹치지 않아야 한다."""
    assert not set(INSTITUTION_COMPARISON_STATE_KEYS) & set(ANALYSIS_STATE_KEYS)


def test_reset_institution_comparison_state_clears_only_institution_keys():
    """기관 비교 초기화는 지정된 기관 비교 키만 비워야 한다."""
    state = filled_institution_state()

    reset_institution_comparison_state(state)

    for key in EXPECTED_INSTITUTION_RESET_KEYS:
        assert state[key] is None


def test_reset_institution_comparison_state_keeps_single_manager_results():
    """기관 비교 초기화가 단일 기관 분석 결과를 지우지 않아야 한다."""
    state = filled_institution_state()

    reset_institution_comparison_state(state)

    for key in EXPECTED_RESET_KEYS:
        assert state[key] is not None
    assert state["active_manager_name"] == "Berkshire Hathaway"
    assert state["selected_filing_index"] == 1


def test_reset_analysis_state_keeps_institution_comparison_results():
    """반대로 단일 기관 초기화도 기관 비교 결과를 건드리지 않아야 한다."""
    state = filled_institution_state()

    reset_analysis_state(state)

    for key in EXPECTED_INSTITUTION_RESET_KEYS:
        assert state[key] is not None


def test_changing_single_manager_keeps_institution_comparison_results():
    """위쪽 기관을 바꿔도 기관 비교 결과는 남아 있어야 한다."""
    state = filled_institution_state()

    sync_selected_manager(state, "Bridgewater Associates")

    for key in EXPECTED_INSTITUTION_RESET_KEYS:
        assert state[key] is not None


@pytest.mark.parametrize(
    "pair",
    [
        ("Bridgewater Associates", "Pershing Square Capital Management"),  # A 변경
        ("Berkshire Hathaway", "Bridgewater Associates"),  # B 변경
    ],
)
def test_changing_institution_pair_resets_only_comparison_results(pair):
    """기관 A 또는 B가 바뀌면 비교 결과만 지워야 한다."""
    state = filled_institution_state()

    changed = sync_institution_pair(state, *pair)

    assert changed is True
    for key in EXPECTED_INSTITUTION_RESET_KEYS:
        assert state[key] is None
    # 단일 기관 분석 결과는 그대로 남습니다.
    for key in EXPECTED_RESET_KEYS:
        assert state[key] is not None
    assert state["active_institution_pair"] == pair


def test_rerun_with_same_institution_pair_keeps_results():
    """같은 기관 조합에서 화면이 다시 실행될 때는 결과가 사라지지 않아야 한다."""
    state = filled_institution_state()
    before = dict(state)

    changed = sync_institution_pair(
        state, "Berkshire Hathaway", "Pershing Square Capital Management"
    )

    assert changed is False
    assert state == before


def test_swapping_institution_sides_resets_results():
    """A와 B를 서로 바꾼 것도 다른 조합이므로 결과를 지워야 한다."""
    state = filled_institution_state()

    changed = sync_institution_pair(
        state, "Pershing Square Capital Management", "Berkshire Hathaway"
    )

    assert changed is True
    assert state["institution_comparison"] is None


# --- 같은 기관 선택 --------------------------------------------------------


def test_same_institution_cannot_be_compared():
    """기관 A와 기관 B가 같으면 비교를 진행하지 않아야 한다."""
    assert can_compare_institutions("Berkshire Hathaway", "Berkshire Hathaway") is False


def test_different_institutions_can_be_compared():
    """서로 다른 두 기관은 비교할 수 있어야 한다."""
    assert (
        can_compare_institutions(
            "Berkshire Hathaway", "Pershing Square Capital Management"
        )
        is True
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("", "Berkshire Hathaway"),
        ("Berkshire Hathaway", ""),
        (None, None),
        ("Berkshire Hathaway", " Berkshire Hathaway "),  # 공백만 다른 같은 이름
    ],
)
def test_incomplete_institution_selection_cannot_be_compared(left, right):
    """기관을 고르지 않았거나 공백만 다른 같은 이름이면 비교하지 않아야 한다."""
    assert can_compare_institutions(left, right) is False


# --- 공통 report_date 찾기 -------------------------------------------------


def test_common_report_dates_are_newest_first():
    """공통 기준일은 최신순으로 나와야 한다(제출일이 아니라 기준일 기준)."""
    both_quarters = LEFT_FILINGS + [
        {
            "accession_number": "0000950123-24-000103",
            "filing_date": "2025-02-10",
            "report_date": "2024-12-31",
            "primary_document": "primary_doc.xml",
        }
    ]

    common = find_common_report_dates(LEFT_FILINGS, both_quarters)

    assert common == ["2025-03-31", "2024-12-31"]


def test_common_report_dates_use_report_date_not_filing_date():
    """제출일이 달라도 기준일이 같으면 공통 분기로 잡아야 한다."""
    common = find_common_report_dates(LEFT_FILINGS, RIGHT_FILINGS)

    assert common == ["2025-03-31"]


def test_no_common_report_date_is_handled_safely():
    """겹치는 분기가 없으면 빈 목록을 돌려주어 화면이 비교를 막을 수 있어야 한다."""
    other_filings = [
        {
            "accession_number": "0000950123-23-000900",
            "filing_date": "2023-08-14",
            "report_date": "2023-06-30",
            "primary_document": "primary_doc.xml",
        }
    ]

    common = find_common_report_dates(LEFT_FILINGS, other_filings)

    assert common == []


def test_institution_filing_for_date_picks_matching_filing():
    """선택한 기준일의 공시(접수번호·제출일)를 기관별로 찾아야 한다."""
    left_by_date = index_filings_by_report_date(LEFT_FILINGS)
    right_by_date = index_filings_by_report_date(RIGHT_FILINGS)

    left_filing = institution_filing_for_date(left_by_date, "2025-03-31")
    right_filing = institution_filing_for_date(right_by_date, "2025-03-31")

    assert left_filing["accession_number"] == "0000950123-25-000001"
    assert left_filing["filing_date"] == "2025-05-15"
    assert right_filing["accession_number"] == "0000950123-25-000101"
    assert right_filing["filing_date"] == "2025-05-12"


@pytest.mark.parametrize(
    ("filings_by_date", "report_date"),
    [
        ({}, "2025-03-31"),  # 아직 조회하지 않은 상태
        (None, "2025-03-31"),  # 상태가 비어 있는 경우
        (index_filings_by_report_date(LEFT_FILINGS), "2019-12-31"),  # 없는 분기
        (index_filings_by_report_date(LEFT_FILINGS), None),  # 기준일 미선택
    ],
)
def test_institution_filing_for_date_returns_empty_when_missing(
    filings_by_date, report_date
):
    """찾지 못하면 빈 딕셔너리를 돌려주어 화면이 멈추지 않아야 한다."""
    assert institution_filing_for_date(filings_by_date, report_date) == {}


# --- 요약 지표 표시 형식 ---------------------------------------------------


def test_format_overlap_percent_uses_one_decimal():
    """중복률·중복도는 소수점 한 자리로 표시하고 부호를 붙이지 않는다."""
    assert format_overlap_percent(12.1948) == "12.2%"
    assert format_overlap_percent(18.5) == "18.5%"
    assert format_overlap_percent(0) == "0.0%"
    assert format_overlap_percent(100) == "100.0%"


def test_format_overlap_percent_handles_missing_value():
    """계산할 수 없는 값은 안내 문구로 바꾼다."""
    assert format_overlap_percent(None) == "계산 불가"
    assert format_overlap_percent(float("nan")) == "계산 불가"


def test_summary_metrics_are_formatted_for_display():
    """요약 지표가 화면 표시용 문자열로 정상 변환되어야 한다."""
    summary = summarize_institution_comparison(sample_institution_comparison())

    # 예시 데이터: 공통 1종목(APPLE), 기관 A 단독 1종목, 기관 B 단독 1종목.
    assert f"{summary['common_count']}개" == "1개"
    assert f"{summary['left_only_count']}개" == "1개"
    assert f"{summary['right_only_count']}개" == "1개"
    # 종목 중복률 = 1 / 3 * 100 = 33.3%
    assert format_overlap_percent(summary["security_overlap_pct"]) == "33.3%"
    # 비중 기준 중복도 = min(60%, 30%) = 30.0%
    assert format_overlap_percent(summary["weighted_overlap_pct"]) == "30.0%"


def test_empty_comparison_summary_is_formatted_safely():
    """비교 결과가 없어도 지표 표시가 오류 없이 0으로 나와야 한다."""
    summary = summarize_institution_comparison(compare_institution_portfolios([], []))

    assert f"{summary['common_count']}개" == "0개"
    assert format_overlap_percent(summary["security_overlap_pct"]) == "0.0%"
    assert format_overlap_percent(summary["weighted_overlap_pct"]) == "0.0%"


def test_short_institution_name_keeps_leading_words():
    """지표·탭 제목에 쓰는 짧은 이름은 앞쪽 단어만 남긴다."""
    assert short_institution_name("Berkshire Hathaway") == "Berkshire"
    assert (
        short_institution_name("Pershing Square Capital Management")
        == "Pershing Square"
    )
    # 짧은 이름은 그대로 씁니다.
    assert short_institution_name("Bridgewater") == "Bridgewater"


def test_short_institution_name_handles_empty_value():
    """이름이 비어 있어도 오류 없이 빈 문자열을 돌려준다."""
    assert short_institution_name(None) == ""
    assert short_institution_name("  ") == ""


# --- 비교 결과 표 ----------------------------------------------------------


def test_institution_display_columns_include_required_columns():
    """비교 결과 표에 필요한 열이 모두 들어 있어야 한다."""
    required = [
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

    for column in required:
        assert column in INSTITUTION_DISPLAY_COLUMNS

    # 내부 식별용 키는 화면에 내보내지 않습니다.
    assert "position_key" not in INSTITUTION_DISPLAY_COLUMNS


def test_institution_display_columns_exist_in_comparison_result():
    """표시할 열이 계산 결과에 실제로 있어야 한다."""
    comparison = sample_institution_comparison()

    for column in INSTITUTION_DISPLAY_COLUMNS:
        assert column in comparison.columns


def test_institution_column_labels_are_korean():
    """표의 열 제목이 한국어 표시명으로 설정되어야 한다."""
    expected_labels = {
        "issuer_name": "종목명",
        "cusip": "CUSIP",
        "put_call": "옵션 구분",
        "share_type": "수량 단위",
        "left_reported_value": "기관 A 공시 평가금액",
        "right_reported_value": "기관 B 공시 평가금액",
        "left_weight_pct": "기관 A 비중",
        "right_weight_pct": "기관 B 비중",
        "weight_gap_pct_point": "비중 차이(%p)",
    }

    config = institution_column_config()

    for column, label in expected_labels.items():
        assert config[column]["label"] == label


def test_institution_rows_by_type_splits_three_groups():
    """공통 보유 / 기관 A 단독 / 기관 B 단독으로 나뉘어야 한다."""
    comparison = sample_institution_comparison()

    common = institution_rows_by_type(comparison, HOLDING_TYPE_COMMON)
    left_only = institution_rows_by_type(comparison, HOLDING_TYPE_LEFT_ONLY)
    right_only = institution_rows_by_type(comparison, HOLDING_TYPE_RIGHT_ONLY)

    assert common["issuer_name"].tolist() == ["APPLE INC"]
    assert left_only["issuer_name"].tolist() == ["COCA COLA CO"]
    assert right_only["issuer_name"].tolist() == ["CHIPOTLE MEXICAN GRILL INC"]


def test_institution_rows_by_type_handles_empty_comparison():
    """비교 결과가 비어 있어도 오류 없이 빈 표를 돌려주어야 한다."""
    empty = compare_institution_portfolios([], [])

    for holding_type in (
        HOLDING_TYPE_COMMON,
        HOLDING_TYPE_LEFT_ONLY,
        HOLDING_TYPE_RIGHT_ONLY,
    ):
        rows = institution_rows_by_type(empty, holding_type)
        assert len(rows) == 0


def test_institution_rows_by_type_handles_missing_column():
    """유형 열이 없는 표를 받아도 빈 표를 돌려주어야 한다."""
    rows = institution_rows_by_type(pd.DataFrame({"issuer_name": ["A"]}), "공통 보유")

    assert len(rows) == 0


# --- dataframe 높이 --------------------------------------------------------


def test_empty_comparison_table_is_not_rendered():
    """비어 있는 비교 결과에서는 st.dataframe을 호출하지 않아야 한다.

    빈 표에 height를 넘겨 StreamlitInvalidHeightError가 나던 문제를 막기 위해,
    행이 없으면 표 대신 안내 문구만 보여 줍니다.
    """
    empty = institution_rows_by_type(
        compare_institution_portfolios([], []), HOLDING_TYPE_COMMON
    )

    with patch.object(streamlit_app, "st", MagicMock()) as fake_st:
        render_institution_holdings_table(empty)

    fake_st.dataframe.assert_not_called()
    fake_st.info.assert_called_once_with("해당 종목이 없습니다.")


@pytest.mark.parametrize("rows", [None, pd.DataFrame()])
def test_missing_comparison_table_is_not_rendered(rows):
    """비교 결과가 아예 없을 때도 표를 그리지 않아야 한다."""
    with patch.object(streamlit_app, "st", MagicMock()) as fake_st:
        render_institution_holdings_table(rows)

    fake_st.dataframe.assert_not_called()


def test_comparison_table_height_is_valid_streamlit_value():
    """표를 그릴 때는 height에 항상 유효한 값(None 아님)을 넘겨야 한다."""
    rows = institution_rows_by_type(sample_institution_comparison(), HOLDING_TYPE_COMMON)

    with patch.object(streamlit_app, "st", MagicMock()) as fake_st:
        render_institution_holdings_table(rows)

    fake_st.dataframe.assert_called_once()
    _assert_valid_streamlit_height(fake_st.dataframe.call_args.kwargs["height"])


def test_comparison_table_height_is_valid_for_many_rows():
    """행이 많은 비교 결과도 유효한 높이(픽셀)를 써서 표 안에서 스크롤되게 한다."""
    many_left = [
        {
            "issuer_name": f"COMPANY {index:03d}",
            "class_title": "COM",
            "cusip": f"{index:09d}",
            "reported_value": 1000 + index,
            "shares": 100,
            "share_type": "SH",
            "put_call": "",
        }
        for index in range(40)
    ]
    rows = institution_rows_by_type(
        compare_institution_portfolios(many_left, many_left), HOLDING_TYPE_COMMON
    )
    assert len(rows) == 40

    with patch.object(streamlit_app, "st", MagicMock()) as fake_st:
        render_institution_holdings_table(rows)

    height = fake_st.dataframe.call_args.kwargs["height"]
    _assert_valid_streamlit_height(height)
    assert height == streamlit_app.LARGE_TABLE_HEIGHT


def test_comparison_table_shows_only_display_columns():
    """표에는 정해진 열만 정해진 순서로 넘겨야 한다."""
    rows = institution_rows_by_type(sample_institution_comparison(), HOLDING_TYPE_COMMON)

    with patch.object(streamlit_app, "st", MagicMock()) as fake_st:
        render_institution_holdings_table(rows)

    rendered = fake_st.dataframe.call_args.args[0]
    assert list(rendered.columns) == INSTITUTION_DISPLAY_COLUMNS


# ---------------------------------------------------------------------------
# 기관 비교 AI 브리핑
#
# 실제 Gemini API에는 요청을 보내지 않습니다. 설정 읽기와 생성 함수를 모두
# 가짜(mock)로 바꿔치기하고, 화면 상태에 무엇이 담기는지만 확인합니다.
# ---------------------------------------------------------------------------

FAKE_GEMINI_API_KEY = "AIza-TEST-FAKE-KEY-1234567890"
FAKE_GEMINI_MODEL = "gemini-test-model"
FAKE_BRIEFING_TEXT = "## 1. 한눈에 보는 비교\n예시 브리핑 본문입니다."

COMPARED_LEFT_NAME = "Berkshire Hathaway"
COMPARED_RIGHT_NAME = "Pershing Square Capital Management"
COMPARED_REPORT_DATE = "2025-03-31"


def compared_state() -> dict:
    """기관 비교를 한 번 마친 뒤의 화면 상태 예시.

    단일 기관 브리핑 결과도 함께 담아 두어, 기관 비교 쪽 동작이 단일 기관
    상태를 건드리지 않는지 확인할 수 있게 합니다.
    """
    comparison = sample_institution_comparison()

    return {
        "institution_comparison": comparison,
        "institution_comparison_summary": summarize_institution_comparison(comparison),
        "institution_selected_report_date": COMPARED_REPORT_DATE,
        "institution_comparison_left_name": COMPARED_LEFT_NAME,
        "institution_comparison_right_name": COMPARED_RIGHT_NAME,
        "institution_ai_briefing": None,
        "institution_ai_briefing_error": None,
        "ai_briefing": "단일 기관 브리핑 본문",
        "ai_briefing_error": None,
    }


def patch_briefing(briefing_text=FAKE_BRIEFING_TEXT, error=None, settings_error=None):
    """AI 브리핑 실행에 필요한 외부 의존을 모두 가짜로 바꿉니다.

    Returns:
        (patcher 목록을 적용하는 컨텍스트 매니저 목록, 가짜 generate_briefing)
    """
    fake_generate = MagicMock()
    if error is not None:
        fake_generate.side_effect = error
    else:
        fake_generate.return_value = briefing_text

    fake_settings = MagicMock()
    if settings_error is not None:
        fake_settings.side_effect = settings_error
    else:
        fake_settings.return_value = (FAKE_GEMINI_API_KEY, FAKE_GEMINI_MODEL)

    return (
        [
            patch.object(streamlit_app, "st", MagicMock()),
            patch.object(streamlit_app, "read_gemini_settings", fake_settings),
            patch.object(streamlit_app, "generate_briefing", fake_generate),
        ],
        fake_generate,
    )


def run_briefing(state, **kwargs):
    """가짜 Gemini로 기관 비교 AI 브리핑을 실행하고 호출 기록을 돌려줍니다."""
    patchers, fake_generate = patch_briefing(**kwargs)

    for patcher in patchers:
        patcher.start()
    try:
        run_institution_ai_briefing(state)
    finally:
        for patcher in reversed(patchers):
            patcher.stop()

    return fake_generate


# --- 상태 키 분리 ----------------------------------------------------------


def test_institution_briefing_state_keys_are_separate_from_single_manager():
    """기관 비교 AI 브리핑 상태 키는 단일 기관 브리핑 키와 겹치지 않아야 한다."""
    assert set(INSTITUTION_AI_BRIEFING_STATE_KEYS) == set(
        EXPECTED_INSTITUTION_BRIEFING_KEYS
    )
    assert not set(INSTITUTION_AI_BRIEFING_STATE_KEYS) & set(ANALYSIS_STATE_KEYS)
    assert "ai_briefing" not in INSTITUTION_AI_BRIEFING_STATE_KEYS
    assert "ai_briefing_error" not in INSTITUTION_AI_BRIEFING_STATE_KEYS


def test_institution_briefing_keys_are_part_of_comparison_state():
    """기관 비교 상태를 지우면 AI 브리핑도 함께 지워지도록 목록에 들어 있어야 한다."""
    for key in INSTITUTION_AI_BRIEFING_STATE_KEYS:
        assert key in INSTITUTION_COMPARISON_STATE_KEYS


def test_reset_institution_comparison_state_clears_ai_briefing():
    """기관 비교 초기화는 기관 비교 AI 브리핑 결과도 지워야 한다."""
    state = filled_institution_state()

    reset_institution_comparison_state(state)

    for key in EXPECTED_INSTITUTION_BRIEFING_KEYS:
        assert state[key] is None


def test_changing_institution_pair_clears_ai_briefing():
    """비교할 기관을 바꾸면 앞서 만든 AI 브리핑이 남아 있지 않아야 한다."""
    state = filled_institution_state()

    changed = sync_institution_pair(state, "Bridgewater Associates", COMPARED_RIGHT_NAME)

    assert changed is True
    for key in EXPECTED_INSTITUTION_BRIEFING_KEYS:
        assert state[key] is None


def test_reset_institution_ai_briefing_keeps_comparison_result():
    """AI 브리핑만 지울 때는 비교 표와 요약 지표를 그대로 두어야 한다."""
    state = filled_institution_state()

    reset_institution_ai_briefing_state(state)

    for key in EXPECTED_INSTITUTION_BRIEFING_KEYS:
        assert state[key] is None
    assert state["institution_comparison"] is not None
    assert state["institution_comparison_summary"] is not None


def test_reset_institution_ai_briefing_keeps_single_manager_briefing():
    """기관 비교 브리핑을 지워도 단일 기관 브리핑은 남아 있어야 한다."""
    state = filled_institution_state()

    reset_institution_ai_briefing_state(state)

    assert state["ai_briefing"] is not None
    assert state["ai_briefing_error"] is not None


def test_changing_report_date_clears_institution_ai_briefing():
    """비교 기준일을 바꾸면 기관 비교 AI 브리핑을 지워야 한다."""
    state = filled_institution_state()
    state["active_institution_briefing_date"] = "2025-03-31"

    changed = sync_institution_briefing_report_date(state, "2024-12-31")

    assert changed is True
    for key in EXPECTED_INSTITUTION_BRIEFING_KEYS:
        assert state[key] is None
    # 비교 결과와 단일 기관 브리핑은 그대로 남아야 합니다.
    assert state["institution_comparison"] is not None
    assert state["ai_briefing"] is not None


def test_same_report_date_keeps_institution_ai_briefing():
    """같은 기준일에서 화면이 다시 그려질 때는 브리핑이 사라지지 않아야 한다."""
    state = compared_state()
    state["active_institution_briefing_date"] = COMPARED_REPORT_DATE
    state["institution_ai_briefing"] = FAKE_BRIEFING_TEXT

    changed = sync_institution_briefing_report_date(state, COMPARED_REPORT_DATE)

    assert changed is False
    assert state["institution_ai_briefing"] == FAKE_BRIEFING_TEXT


# --- AI 입력 데이터 만들기 -------------------------------------------------


def test_briefing_payload_uses_only_comparison_state():
    """화면 상태의 비교 결과에서 정해진 값만 AI 입력 데이터로 옮겨야 한다."""
    payload = institution_briefing_payload_from_state(compared_state())

    assert payload["report_date"] == COMPARED_REPORT_DATE
    assert payload["left_manager_name"] == COMPARED_LEFT_NAME
    assert payload["right_manager_name"] == COMPARED_RIGHT_NAME
    assert payload["summary"]["common_count"] == 1  # APPLE INC
    assert [row["issuer_name"] for row in payload["top_common_holdings"]] == [
        "APPLE INC"
    ]


def test_briefing_payload_handles_empty_state():
    """비교 결과가 없는 상태에서도 오류 없이 입력 데이터를 만들어야 한다."""
    payload = institution_briefing_payload_from_state({})

    assert payload["report_date"] == ""
    assert payload["left_manager_name"] == "기관 A"
    assert payload["top_common_holdings"] == []


# --- Gemini 호출 (mock) ----------------------------------------------------


def test_briefing_result_is_stored_in_institution_state():
    """생성 결과는 기관 비교 전용 상태에 담겨야 한다."""
    state = compared_state()

    fake_generate = run_briefing(state)

    assert state["institution_ai_briefing"] == FAKE_BRIEFING_TEXT
    assert state["institution_ai_briefing_error"] is None
    fake_generate.assert_called_once()

    # 프롬프트는 문자열이고, API 키와 모델명은 인자로만 넘깁니다.
    prompt = fake_generate.call_args.args[0]
    assert isinstance(prompt, str)
    assert fake_generate.call_args.kwargs["api_key"] == FAKE_GEMINI_API_KEY
    assert fake_generate.call_args.kwargs["model_name"] == FAKE_GEMINI_MODEL


def test_briefing_prompt_contains_comparison_values_only():
    """Gemini에 넘기는 프롬프트에 전체 비교 표나 내부 키가 없어야 한다."""
    state = compared_state()

    fake_generate = run_briefing(state)
    prompt = fake_generate.call_args.args[0]

    assert COMPARED_REPORT_DATE in prompt
    assert COMPARED_LEFT_NAME in prompt
    assert "position_key" not in prompt
    assert "informationTable" not in prompt


def test_briefing_does_not_touch_single_manager_state():
    """기관 비교 브리핑을 만들어도 단일 기관 브리핑 상태는 그대로여야 한다."""
    state = compared_state()

    run_briefing(state)

    assert state["ai_briefing"] == "단일 기관 브리핑 본문"


def test_briefing_api_error_is_stored_in_institution_error_state():
    """Gemini 호출 오류는 기관 비교 전용 오류 상태에 담겨야 한다."""
    state = compared_state()

    run_briefing(state, error=LlmApiError("Gemini API 사용량 한도를 넘었습니다."))

    assert state["institution_ai_briefing"] is None
    assert "사용량 한도" in state["institution_ai_briefing_error"]
    # 단일 기관 브리핑 오류 상태에는 담기지 않아야 합니다.
    assert state["ai_briefing_error"] is None


def test_briefing_missing_settings_error_is_stored():
    """GEMINI 설정이 없을 때의 안내도 기관 비교 오류 상태에 담겨야 한다."""
    state = compared_state()

    run_briefing(state, settings_error=LookupError("GEMINI_API_KEY가 없습니다."))

    assert state["institution_ai_briefing"] is None
    assert "GEMINI_API_KEY" in state["institution_ai_briefing_error"]


def test_briefing_unexpected_error_does_not_leak_details():
    """예상하지 못한 오류에서는 원본 내용을 화면 상태에 담지 않아야 한다."""
    state = compared_state()

    run_briefing(state, error=RuntimeError(f"unexpected {FAKE_GEMINI_API_KEY}"))

    assert state["institution_ai_briefing"] is None
    assert (
        state["institution_ai_briefing_error"]
        == streamlit_app.INSTITUTION_BRIEFING_UNEXPECTED_ERROR
    )
    assert FAKE_GEMINI_API_KEY not in state["institution_ai_briefing_error"]


def test_briefing_error_is_cleared_on_retry():
    """다시 생성해서 성공하면 앞선 오류 문구가 남지 않아야 한다."""
    state = compared_state()
    state["institution_ai_briefing_error"] = "이전 오류 문구"

    run_briefing(state)

    assert state["institution_ai_briefing_error"] is None
    assert state["institution_ai_briefing"] == FAKE_BRIEFING_TEXT


def test_briefing_calls_api_once_per_button_click():
    """한 번 실행에 Gemini 호출은 한 번이어야 한다(화면 rerun으로 재호출 없음)."""
    state = compared_state()

    fake_generate = run_briefing(state)

    assert fake_generate.call_count == 1

    # 화면이 다시 그려지는 것만으로는 호출되지 않고, 상태에 담긴 글을 다시 보여줍니다.
    assert state["institution_ai_briefing"] == FAKE_BRIEFING_TEXT


def test_briefing_notice_mentions_no_investment_advice():
    """화면 안내 문구에 투자 추천·투자 의도 추정을 하지 않는다는 내용이 있어야 한다."""
    notice = streamlit_app.INSTITUTION_BRIEFING_NOTICE

    assert "투자 추천" in notice
    assert "투자 의도" in notice
    assert "Python으로 계산된 기관 비교 결과" in notice


def test_briefing_spinner_text_is_fixed():
    """브리핑 생성 중 문구는 정해진 문장이어야 한다."""
    assert (
        streamlit_app.INSTITUTION_BRIEFING_SPINNER_TEXT
        == "기관 비교 결과를 바탕으로 AI 브리핑을 생성하고 있습니다."
    )


# ---------------------------------------------------------------------------
# 화면 디자인 (Hero 영역과 컬러 시스템)
#
# 보이는 모습만 확인합니다. 이 테스트들은 SEC나 Gemini에 접속하지 않습니다.
# ---------------------------------------------------------------------------

# 화면에 쓰기로 정한 여섯 가지 색.
EXPECTED_PALETTE = {
    "COLOR_BACKGROUND": "#D4CFC2",
    "COLOR_HEADING": "#00533E",
    "COLOR_BODY": "#000000",
    "COLOR_ACCENT": "#F7633D",
    "COLOR_SECONDARY": "#A35D3F",
    "COLOR_BADGE": "#ECB97A",
}

# 쓰지 않기로 한 계열(파랑·보라·형광). 차트 전용 색은 아래에서 따로 확인합니다.
FORBIDDEN_HUES = ["#0000ff", "#00f", "purple", "violet", "magenta", "#39ff14", "#0ff"]


def tiny_jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    """테스트용 아주 작은 JPEG 이미지를 만듭니다(파일을 저장소에 추가하지 않기 위함)."""
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(180, 170, 150)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


SMALLEST_JPEG = tiny_jpeg_bytes()


@pytest.fixture(autouse=True)
def empty_hero_image_cache():
    """Hero 이미지 캐시를 테스트마다 비워 다른 경로 결과가 섞이지 않게 합니다."""
    hero_image_data_uri.clear()
    yield
    hero_image_data_uri.clear()


# --- 컬러 시스템 -----------------------------------------------------------


@pytest.mark.parametrize(("name", "value"), sorted(EXPECTED_PALETTE.items()))
def test_palette_uses_exact_specified_colors(name, value):
    """지정한 여섯 가지 색을 정확한 값으로 써야 한다."""
    assert getattr(streamlit_app, name) == value


def test_theme_variables_declare_every_palette_color():
    """CSS 변수 선언에 여섯 가지 색이 모두 들어 있어야 한다."""
    variables = theme_variables_css()

    for value in EXPECTED_PALETTE.values():
        assert value in variables


def test_page_style_does_not_add_blue_or_neon_colors():
    """화면 스타일에 임의의 파랑·보라·형광색을 넣지 않아야 한다."""
    stylesheet = (theme_variables_css() + streamlit_app.PAGE_STYLE_CSS).lower()

    for hue in FORBIDDEN_HUES:
        assert hue not in stylesheet


def test_chart_colors_are_unchanged():
    """비중 변화 차트의 기존 색은 이번 디자인 변경 대상이 아니다."""
    assert streamlit_app.COLOR_INCREASE == "#e34948"
    assert streamlit_app.COLOR_DECREASE == "#2a78d6"


def test_page_style_uses_testid_selectors():
    """Streamlit 내부 클래스 대신 data-testid 기준으로 스타일을 잡아야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS

    for selector in (
        '[data-testid="stAppViewContainer"]',
        '[data-testid="stHeader"]',
        '[data-testid="stMainBlockContainer"]',
        '[data-testid="stBaseButton-primary"]',
        '[data-testid="stSelectbox"]',
        '[data-testid="stTab"]',
        '[data-testid="stMetric"]',
    ):
        assert selector in stylesheet

    # 버전마다 바뀌는 자동 생성 클래스에 기대지 않아야 합니다.
    assert "st-emotion-cache" not in stylesheet


def test_page_background_matches_header_background():
    """상단 영역과 본문 배경이 같은 색으로 통일되어야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS
    background_rule = stylesheet.split("[data-testid=\"stHeader\"] {")[0]

    assert '[data-testid="stHeader"]' in background_rule
    assert '[data-testid="stMain"]' in background_rule
    assert "background-color: var(--f13-bg)" in background_rule


def test_content_width_is_limited_even_in_wide_layout():
    """layout='wide'에서도 본문이 지나치게 넓어지지 않도록 폭을 제한한다."""
    assert 900 <= streamlit_app.CONTENT_MAX_WIDTH <= 1400
    assert "max-width: var(--f13-content-width)" in streamlit_app.PAGE_STYLE_CSS


# --- Hero 문구와 타이포그래피 ----------------------------------------------


def test_hero_shows_required_texts():
    """Hero에 지정한 라벨·제목·보조 설명이 그대로 들어가야 한다."""
    html = hero_html()

    assert "SEC 13F INTELLIGENCE PLATFORM" in html
    assert "AI 13F Portfolio Analysis" in html
    assert (
        "Analyze institutional filings, portfolio changes, and comparative "
        "positioning with AI-assisted insights." in html
    )


def test_hero_keeps_poc_wording_as_small_note_only():
    """PoC 표현은 메인 제목이 아니라 작은 보조 텍스트로만 남아야 한다."""
    assert "PoC" not in streamlit_app.HERO_TITLE
    assert "PoC" in streamlit_app.HERO_NOTE


def test_hero_does_not_reference_external_urls():
    """Hero는 외부 이미지 주소를 쓰지 않아야 한다."""
    combined = hero_html() + streamlit_app.PAGE_STYLE_CSS + hero_background_css()

    assert "http://" not in combined
    assert "https://" not in combined


def test_hero_title_uses_safe_serif_stack():
    """메인 영문 제목은 Georgia 우선의 안전한 serif 조합을 써야 한다."""
    assert (
        'font-family: Georgia, "Times New Roman", serif'
        in streamlit_app.PAGE_STYLE_CSS
    )
    # 저장소에 폰트 파일을 추가하지 않았는지도 함께 확인합니다.
    assert "@font-face" not in streamlit_app.PAGE_STYLE_CSS


def test_hero_title_size_shrinks_on_narrow_screens():
    """제목 크기는 clamp()로 화면 폭에 따라 줄어들어야 한다."""
    assert "font-size: clamp(2.4rem, 5vw, 4.25rem)" in streamlit_app.PAGE_STYLE_CSS


def test_hero_height_is_in_requested_range():
    """Hero 높이는 요청한 340~400px 범위 안이어야 한다."""
    assert "min-height: 360px" in streamlit_app.PAGE_STYLE_CSS


# --- Hero 배경 이미지와 fallback -------------------------------------------


def test_hero_image_path_points_to_assets_folder():
    """Hero 이미지 경로는 assets/wall_street_hero.jpg여야 한다."""
    assert streamlit_app.HERO_IMAGE_PATH.name == "wall_street_hero.jpg"
    assert streamlit_app.HERO_IMAGE_PATH.parent.name == "assets"


def test_hero_uses_local_image_as_data_uri_when_available(tmp_path):
    """이미지가 있으면 base64 data URI로 읽어 CSS 배경에 넣어야 한다."""
    image_file = tmp_path / "wall_street_hero.jpg"
    # 1x1 픽셀 JPEG(가장 작은 유효 이미지)로 실제 읽기 경로를 확인합니다.
    image_file.write_bytes(SMALLEST_JPEG)

    css = hero_background_css(image_file)

    assert 'background-image: url("data:image/jpeg;base64,' in css
    assert streamlit_app.HERO_IMAGE_FILTER in css


def test_hero_falls_back_when_image_is_missing(tmp_path):
    """이미지 파일이 없어도 앱이 멈추지 않고 fallback 배경을 써야 한다."""
    css = hero_background_css(tmp_path / "없는파일.jpg")

    assert streamlit_app.HERO_FALLBACK_BACKGROUND in css
    assert "data:image" not in css


def test_hero_falls_back_when_image_cannot_be_read(tmp_path):
    """경로가 폴더처럼 읽을 수 없는 경우에도 오류 없이 fallback을 써야 한다."""
    unreadable = tmp_path / "폴더입니다"
    unreadable.mkdir()

    css = hero_background_css(unreadable)

    assert streamlit_app.HERO_FALLBACK_BACKGROUND in css


def test_hero_falls_back_for_empty_image_file(tmp_path):
    """빈 파일이어도 오류 없이 fallback 배경을 써야 한다."""
    empty_file = tmp_path / "wall_street_hero.jpg"
    empty_file.write_bytes(b"")

    assert streamlit_app.HERO_FALLBACK_BACKGROUND in hero_background_css(empty_file)


def test_hero_fallback_uses_only_specified_colors():
    """fallback 그라데이션은 #D4CFC2와 #00533E만 써야 한다."""
    fallback = streamlit_app.HERO_FALLBACK_BACKGROUND

    assert "#D4CFC2" in fallback
    assert "0, 83, 62" in fallback  # #00533E의 rgb 값
    for hue in FORBIDDEN_HUES:
        assert hue not in fallback.lower()


def test_hero_overlay_is_light_beige_not_black():
    """이미지 위 오버레이는 검정이 아니라 밝은 베이지여야 한다."""
    overlay = streamlit_app.HERO_OVERLAY

    assert "rgba(212, 207, 194" in overlay  # #D4CFC2
    assert "rgba(0, 0, 0" not in overlay


def test_hero_background_is_used_only_inside_hero():
    """배경 이미지는 Hero 영역에만 쓰고 표·본문에는 쓰지 않아야 한다."""
    assert hero_background_css().startswith(".f13-hero::before{")
    assert "background-image: url(" not in streamlit_app.PAGE_STYLE_CSS


@pytest.mark.parametrize("size", [(4000, 2000), (3064, 4592), (400, 300)])
def test_hero_image_is_resized_to_wide_hero_shape(size):
    """가로·세로 어느 쪽이 길든 Hero 크기에 맞춰 잘라 줄여야 한다."""
    pillow_image = pytest.importorskip("PIL.Image")

    buffer = BytesIO()
    pillow_image.new("RGB", size, color=(120, 120, 120)).save(
        buffer, format="JPEG", quality=95
    )

    compacted = streamlit_app.compact_hero_image(buffer.getvalue())

    with pillow_image.open(BytesIO(compacted)) as reduced:
        assert reduced.size == (
            streamlit_app.HERO_IMAGE_MAX_WIDTH,
            streamlit_app.HERO_IMAGE_MAX_HEIGHT,
        )


def test_large_hero_image_becomes_much_smaller():
    """큰 원본을 그대로 싣지 않고 충분히 줄여야 한다(화면 갱신 속도)."""
    pillow_image = pytest.importorskip("PIL.Image")

    buffer = BytesIO()
    pillow_image.new("RGB", (3064, 4592), color=(120, 130, 140)).save(
        buffer, format="JPEG", quality=95
    )
    original = buffer.getvalue()

    assert len(streamlit_app.compact_hero_image(original)) < len(original)


def test_broken_image_bytes_are_kept_as_is():
    """이미지를 줄이지 못해도 오류 없이 원본을 그대로 돌려주어야 한다."""
    broken = b"not really an image"

    assert streamlit_app.compact_hero_image(broken) == broken


# --- 기존 기능 연결 유지 ---------------------------------------------------


def test_page_layout_is_wide():
    """set_page_config의 layout이 wide로 설정되어야 한다."""
    source = (Path(streamlit_app.__file__)).read_text(encoding="utf-8")

    assert 'layout="wide"' in source


def test_streamlit_title_is_replaced_by_hero():
    """기존 st.title은 Hero 제목으로 대체되어 중복 표시되지 않아야 한다."""
    source = (Path(streamlit_app.__file__)).read_text(encoding="utf-8")

    assert "st.title(" not in source
    assert "render_hero()" in source


def test_existing_button_keys_are_preserved():
    """디자인 변경 뒤에도 기존 버튼 key가 그대로 남아 있어야 한다."""
    source = (Path(streamlit_app.__file__)).read_text(encoding="utf-8")

    for key in (
        'key="institution_common_dates_button"',
        'key="institution_compare_button"',
        'key="institution_ai_briefing_button"',
    ):
        assert key in source


def test_existing_selectbox_keys_are_preserved():
    """디자인 변경 뒤에도 기존 selectbox key가 그대로 남아 있어야 한다."""
    source = (Path(streamlit_app.__file__)).read_text(encoding="utf-8")

    for key in (
        'key="selected_manager_name"',
        'key="selected_filing_index"',
        'key="institution_left_manager"',
        'key="institution_right_manager"',
        'key="institution_report_date_choice"',
    ):
        assert key in source


def test_analysis_functions_are_still_called_from_services():
    """분석·조회 함수는 services 모듈의 것을 그대로 써야 한다(복제 금지)."""
    for name, module in (
        ("compare_holdings", "services.portfolio_analysis"),
        ("summarize_comparison", "services.portfolio_analysis"),
        ("compare_institution_portfolios", "services.institution_comparison"),
        ("summarize_institution_comparison", "services.institution_comparison"),
        ("get_recent_13f_filings", "services.sec_client"),
        ("get_13f_holdings", "services.sec_client"),
        ("generate_briefing", "services.llm_client"),
    ):
        assert getattr(streamlit_app, name).__module__ == module


# ---------------------------------------------------------------------------
# 표 디자인 (검은색 기반 다크 테이블)
#
# st.dataframe의 열 헤더와 칸은 <canvas>(그림판)에 그려져 CSS가 닿지 않습니다.
# 그래서 표 색은 `.streamlit/config.toml`의 [theme]에서 정하고,
# 아래 테스트로 파이썬 상수와 그 파일이 어긋나지 않았는지 확인합니다.
# ---------------------------------------------------------------------------

# 표에 쓰기로 정한 다크 팔레트.
EXPECTED_TABLE_PALETTE = {
    "COLOR_TABLE_HEADER_BG": "#000000",
    "COLOR_TABLE_BG": "#0F0F0F",
    "COLOR_TABLE_TEXT": "#FFFFFF",
    "COLOR_TABLE_MUTED_TEXT": "#D4CFC2",
    "COLOR_TABLE_BORDER": "#593627",
    "COLOR_TABLE_HOVER": "#30281F",
}


def read_streamlit_theme() -> dict:
    """`.streamlit/config.toml`의 [theme] 부분만 읽어 옵니다.

    비밀 정보가 담긴 secrets.toml은 건드리지 않습니다.
    """
    import tomllib

    config_path = Path(streamlit_app.__file__).parent / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)["theme"]


@pytest.mark.parametrize(("name", "value"), sorted(EXPECTED_TABLE_PALETTE.items()))
def test_table_palette_uses_exact_specified_colors(name, value):
    """표에 쓰기로 정한 어두운 색을 정확한 값으로 써야 한다."""
    assert getattr(streamlit_app, name) == value


def test_theme_variables_declare_every_table_color():
    """CSS 변수 선언에 표 전용 색이 모두 들어 있어야 한다."""
    variables = theme_variables_css()

    for value in EXPECTED_TABLE_PALETTE.values():
        assert value in variables


def test_streamlit_theme_paints_dataframe_dark():
    """설정 파일의 [theme]이 표를 검은색 기반 다크로 그리게 해야 한다."""
    theme = read_streamlit_theme()

    assert theme["dataframeHeaderBackgroundColor"] == streamlit_app.COLOR_TABLE_HEADER_BG
    assert theme["backgroundColor"] == streamlit_app.COLOR_TABLE_BG
    assert theme["textColor"] == streamlit_app.COLOR_TABLE_TEXT
    assert theme["dataframeBorderColor"] == streamlit_app.COLOR_TABLE_BORDER
    assert theme["secondaryBackgroundColor"] == streamlit_app.COLOR_TABLE_HOVER_SOURCE
    assert theme["base"] == "dark"


def test_table_header_text_is_bold_enough():
    """열 헤더 글씨는 600 이상으로 굵게 그려야 한다."""
    assert streamlit_app.TABLE_FONT_WEIGHT >= 600
    assert read_streamlit_theme()["baseFontWeight"] == streamlit_app.TABLE_FONT_WEIGHT


def test_page_restores_normal_font_weight_outside_tables():
    """표 밖의 본문 글씨는 예전처럼 보통 굵기로 되돌려야 한다."""
    assert streamlit_app.BODY_FONT_WEIGHT == 400
    assert "font-weight: var(--f13-body-weight)" in streamlit_app.PAGE_STYLE_CSS


def test_dataframe_container_matches_dark_table():
    """표를 감싸는 상자도 어두운 배경과 따뜻한 구분선을 써야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS
    block = stylesheet.split('[data-testid="stDataFrame"] {')[1].split("}")[0]

    assert "background-color: var(--f13-table-bg)" in block
    assert "border: 1px solid var(--f13-table-border)" in block


def test_html_tables_are_dark_too():
    """st.table과 Markdown 표도 st.dataframe과 같은 다크 테이블이어야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS

    for selector in (
        '[data-testid="stTable"] thead th',
        '[data-testid="stMarkdown"] thead th',
        '[data-testid="stTable"] tbody td',
        '[data-testid="stMarkdown"] tbody td',
    ):
        assert selector in stylesheet

    assert "background-color: var(--f13-table-header-bg)" in stylesheet
    assert "background-color: var(--f13-table-hover)" in stylesheet


def test_table_toolbar_icons_stay_visible_on_dark_table():
    """어두운 표 위 도구 모음 아이콘이 검정이라 안 보이는 일이 없어야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS
    block = stylesheet.split('[data-testid="stElementToolbar"] button,')[1]
    block = block.split("}")[0]

    assert "color: var(--f13-table-text)" in block


def test_site_keeps_light_beige_page_despite_dark_theme():
    """표만 어둡고, 사이트 배경·제목·본문 색은 그대로여야 한다."""
    assert streamlit_app.COLOR_BACKGROUND == "#D4CFC2"
    assert streamlit_app.COLOR_HEADING == "#00533E"
    assert streamlit_app.COLOR_BODY == "#000000"

    stylesheet = streamlit_app.PAGE_STYLE_CSS

    assert "background-color: var(--f13-bg)" in stylesheet
    assert "color: var(--f13-heading)" in stylesheet


def test_selectbox_arrow_stays_white():
    """드롭다운 화살표 흰색 처리는 그대로 남아 있어야 한다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS
    block = stylesheet.split('[data-testid="stSelectbox"] svg,')[1].split("}")[0]

    assert "fill: #FFFFFF" in block


def test_ai_briefing_card_design_is_unchanged():
    """AI 브리핑 카드 디자인은 이번 표 변경 대상이 아니다."""
    stylesheet = streamlit_app.PAGE_STYLE_CSS

    assert ".st-key-ai_briefing_card" in stylesheet
    assert ".st-key-institution_ai_briefing_card" in stylesheet
    assert "background-color: rgba(var(--f13-badge-rgb), 0.2)" in stylesheet


def test_hero_note_and_description_are_unchanged():
    """상단 PoC 문구와 13F 한 줄 설명은 그대로여야 한다."""
    assert streamlit_app.HERO_NOTE == "SEC EDGAR 13F-HR 공시 기반 PoC"
    assert streamlit_app.HERO_DESCRIPTION == (
        "13F는 미국 기관투자자가 SEC에 분기별로 제출하는 보유 주식 공시입니다."
    )


def test_chart_text_stays_dark_on_light_page():
    """다크 테마를 쓰더라도 차트 글씨는 밝은 배경용 색이어야 한다."""
    chart_config = streamlit_app.weight_change_chart(
        pd.DataFrame(
            {
                "issuer_name": ["가", "나"],
                "weight_change_pct_point": [1.5, -2.5],
                "change_status": ["비중 확대", "비중 축소"],
                "변화 방향": ["비중 확대", "비중 축소"],
            }
        )
    ).to_dict()["config"]

    assert chart_config["axis"]["labelColor"] == streamlit_app.CHART_LABEL_COLOR
    assert chart_config["legend"]["labelColor"] == streamlit_app.CHART_LABEL_COLOR
    assert chart_config["background"] == "transparent"
