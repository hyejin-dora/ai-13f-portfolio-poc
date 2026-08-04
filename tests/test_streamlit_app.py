"""streamlit_app.py의 기관투자자 선택 관련 기능 테스트.

실제 화면을 띄우지 않고, 운용사 목록 읽기와 선택 상태 처리만 확인합니다.
Streamlit 위젯은 `streamlit run` 없이 불러오면 기본값을 돌려주므로,
이 파일은 모듈을 그대로 import 해서 안에 있는 함수만 검증합니다.
"""

import pandas as pd
import pytest

import streamlit_app
from streamlit_app import (
    ANALYSIS_STATE_KEYS,
    DEFAULT_MANAGER,
    default_manager_index,
    find_manager,
    load_managers,
    manager_options,
    reset_analysis_state,
    sync_selected_manager,
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
