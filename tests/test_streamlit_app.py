"""streamlit_app.py의 기관투자자 선택 및 SEC 조회 캐시 테스트.

실제 화면을 띄우지 않고, 운용사 목록 읽기와 선택 상태 처리,
그리고 SEC 조회 캐시 래퍼의 동작만 확인합니다.
Streamlit 위젯은 `streamlit run` 없이 불러오면 기본값을 돌려주므로,
이 파일은 모듈을 그대로 import 해서 안에 있는 함수만 검증합니다.

SEC에 실제로 접속하지 않습니다. 조회 함수는 모두 가짜(mock)로 바꿔치기합니다.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import streamlit_app
from streamlit_app import (
    ANALYSIS_STATE_KEYS,
    DEFAULT_MANAGER,
    cached_13f_holdings,
    cached_recent_filings,
    clear_sec_caches,
    default_manager_index,
    find_manager,
    load_managers,
    manager_options,
    reset_analysis_state,
    sync_selected_manager,
    table_height,
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
