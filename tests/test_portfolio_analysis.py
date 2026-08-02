"""services/portfolio_analysis.py 테스트.

외부 API를 호출하지 않는 순수 계산 모듈이므로 네트워크 요청이나 mock이 필요 없습니다.
예시 보유 종목 목록을 직접 만들어 계산 결과를 검증합니다.
"""

import copy

import pandas as pd
import pytest

from services.portfolio_analysis import (
    COMPARISON_COLUMNS,
    STATUS_DECREASED,
    STATUS_EXITED,
    STATUS_INCREASED,
    STATUS_NEW,
    STATUS_UNCHANGED,
    aggregate_holdings,
    compare_holdings,
    summarize_comparison,
)

# ---------------------------------------------------------------------------
# 예시 데이터
#
# 이전 분기 합계: 1000 + 500 + 300 + (100 + 200) = 2100
# 현재 분기 합계: 1500 + 400 + 300 + 700       = 2900
#
# 종목별로 확인하려는 변화:
#   APPLE        보유수량 100 -> 120  : 보유 확대
#   COCA COLA    보유수량  50 ->  40  : 보유 축소
#   WELLS FARGO  두 줄로 나뉜 공시(10 + 20 = 30)가 그대로 30  : 유지
#   BANK OF AM.  현재 분기에 없음                            : 전량 매도
#   OCCIDENTAL   이전 분기에 없음                            : 신규 편입
# ---------------------------------------------------------------------------

PREVIOUS_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "value_thousands": 1000,
        "shares": 100,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "value_thousands": 500,
        "shares": 50,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "BANK OF AMERICA CORP",
        "class_title": "COM",
        "cusip": "060505104",
        "value_thousands": 300,
        "shares": 30,
        "share_type": "SH",
        "put_call": "",
    },
    # 같은 CUSIP이 두 줄로 나뉘어 공시된 경우(합산 대상).
    {
        "issuer_name": "WELLS FARGO & CO",
        "class_title": "COM",
        "cusip": "949746101",
        "value_thousands": 100,
        "shares": 10,
        "share_type": "SH",
        "put_call": "",
    },
    {
        # 발행사명과 증권 종류가 비어 있어도, 다른 줄의 값을 쓰면 됩니다.
        "issuer_name": "",
        "class_title": "",
        "cusip": "949746101",
        "value_thousands": 200,
        "shares": 20,
        "share_type": "SH",
        "put_call": "",
    },
]

CURRENT_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "value_thousands": 1500,
        "shares": 120,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "value_thousands": 400,
        "shares": 40,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "WELLS FARGO & CO",
        "class_title": "COM",
        "cusip": "949746101",
        "value_thousands": 300,
        "shares": 30,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "OCCIDENTAL PETROLEUM CORP",
        "class_title": "COM",
        "cusip": "674599105",
        "value_thousands": 700,
        "shares": 70,
        "share_type": "SH",
        "put_call": "",
    },
]

PREVIOUS_TOTAL = 2100
CURRENT_TOTAL = 2900


@pytest.fixture
def comparison() -> pd.DataFrame:
    """예시 데이터로 만든 비교 결과 표."""
    return compare_holdings(PREVIOUS_HOLDINGS, CURRENT_HOLDINGS)


def row_of(comparison: pd.DataFrame, cusip: str) -> pd.Series:
    """CUSIP으로 한 줄을 찾아 돌려주는 테스트 보조 함수."""
    matched = comparison[comparison["cusip"] == cusip]
    assert len(matched) == 1, f"CUSIP {cusip} 줄이 정확히 1개여야 합니다."
    return matched.iloc[0]


# ---------------------------------------------------------------------------
# 1. 신규 편입 종목 분류
# ---------------------------------------------------------------------------


def test_classifies_new_position(comparison):
    """이전 분기에 없던 종목은 '신규 편입'으로 분류됩니다."""
    occidental = row_of(comparison, "674599105")

    assert occidental["change_status"] == STATUS_NEW
    assert occidental["issuer_name"] == "OCCIDENTAL PETROLEUM CORP"
    assert occidental["previous_value_thousands"] == 0
    assert occidental["previous_shares"] == 0
    assert occidental["current_value_thousands"] == 700
    assert occidental["current_shares"] == 70
    assert occidental["value_change_thousands"] == 700
    assert occidental["shares_change"] == 70


# ---------------------------------------------------------------------------
# 2. 전량 매도 종목 분류
# ---------------------------------------------------------------------------


def test_classifies_exited_position(comparison):
    """현재 분기에 사라진 종목은 '전량 매도'로 분류되고 결과에 남아 있습니다."""
    bank_of_america = row_of(comparison, "060505104")

    assert bank_of_america["change_status"] == STATUS_EXITED
    # 이전 분기 공시에만 있으므로 이름은 이전 공시 표기를 씁니다.
    assert bank_of_america["issuer_name"] == "BANK OF AMERICA CORP"
    assert bank_of_america["previous_value_thousands"] == 300
    assert bank_of_america["current_value_thousands"] == 0
    assert bank_of_america["current_shares"] == 0
    assert bank_of_america["value_change_thousands"] == -300
    assert bank_of_america["shares_change"] == -30
    # 전량 매도이므로 변화율은 -100%입니다.
    assert bank_of_america["value_change_pct"] == pytest.approx(-100.0)
    assert bank_of_america["current_weight"] == 0


def test_exited_position_is_included_in_result(comparison):
    """한쪽에만 있는 종목까지 모두 남는지(outer join) 확인합니다."""
    # 이전 4종목 + 현재 4종목 중 3종목이 겹치므로 전체는 5줄입니다.
    assert len(comparison) == 5

    cusips = set(comparison["cusip"])
    assert cusips == {
        "037833100",  # APPLE (유지 대상 아님, 확대)
        "191216100",  # COCA COLA
        "949746101",  # WELLS FARGO
        "060505104",  # BANK OF AMERICA (전량 매도)
        "674599105",  # OCCIDENTAL (신규 편입)
    }


# ---------------------------------------------------------------------------
# 3. 보유 확대 / 보유 축소 분류
# ---------------------------------------------------------------------------


def test_classifies_increased_position(comparison):
    """보유수량이 늘어난 종목은 '보유 확대'로 분류됩니다."""
    apple = row_of(comparison, "037833100")

    assert apple["change_status"] == STATUS_INCREASED
    assert apple["previous_shares"] == 100
    assert apple["current_shares"] == 120
    assert apple["shares_change"] == 20
    assert apple["value_change_thousands"] == 500
    assert apple["value_change_pct"] == pytest.approx(50.0)


def test_classifies_decreased_position(comparison):
    """보유수량이 줄어든 종목은 '보유 축소'로 분류됩니다."""
    coca_cola = row_of(comparison, "191216100")

    assert coca_cola["change_status"] == STATUS_DECREASED
    assert coca_cola["previous_shares"] == 50
    assert coca_cola["current_shares"] == 40
    assert coca_cola["shares_change"] == -10
    assert coca_cola["value_change_thousands"] == -100
    assert coca_cola["value_change_pct"] == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# 4. 유지 종목 분류
# ---------------------------------------------------------------------------


def test_classifies_unchanged_position(comparison):
    """보유수량이 그대로인 종목은 '유지'로 분류됩니다."""
    wells_fargo = row_of(comparison, "949746101")

    assert wells_fargo["change_status"] == STATUS_UNCHANGED
    assert wells_fargo["previous_shares"] == 30
    assert wells_fargo["current_shares"] == 30
    assert wells_fargo["shares_change"] == 0


# ---------------------------------------------------------------------------
# 5. 동일 CUSIP 중복 행 합산
# ---------------------------------------------------------------------------


def test_aggregate_holdings_sums_duplicate_cusip_rows():
    """같은 CUSIP이 여러 줄이면 평가금액과 보유수량을 더합니다."""
    aggregated = aggregate_holdings(PREVIOUS_HOLDINGS)

    # 5줄이 4종목으로 합쳐집니다.
    assert len(aggregated) == 4

    wells_fargo = aggregated[aggregated["cusip"] == "949746101"].iloc[0]
    assert wells_fargo["value_thousands"] == 300  # 100 + 200
    assert wells_fargo["shares"] == 30  # 10 + 20
    # 비어 있는 값 대신, 비어 있지 않은 첫 번째 값을 씁니다.
    assert wells_fargo["issuer_name"] == "WELLS FARGO & CO"
    assert wells_fargo["class_title"] == "COM"


def test_compare_holdings_uses_summed_duplicate_rows(comparison):
    """비교 결과에도 합산된 값이 반영되는지 확인합니다."""
    wells_fargo = row_of(comparison, "949746101")

    assert wells_fargo["previous_value_thousands"] == 300
    assert wells_fargo["previous_shares"] == 30


def test_aggregate_holdings_uses_first_valid_issuer_name():
    """앞쪽 줄의 이름이 비어 있으면 뒤쪽 줄의 이름을 사용합니다."""
    holdings = [
        {"cusip": "111111111", "issuer_name": "", "class_title": "", "value_thousands": 10, "shares": 1},
        {"cusip": "111111111", "issuer_name": "SECOND ROW CO", "class_title": "COM", "value_thousands": 20, "shares": 2},
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["issuer_name"] == "SECOND ROW CO"
    assert aggregated.iloc[0]["class_title"] == "COM"
    assert aggregated.iloc[0]["value_thousands"] == 30


# ---------------------------------------------------------------------------
# 6. 이전 / 현재 포트폴리오 비중 계산
# ---------------------------------------------------------------------------


def test_calculates_previous_and_current_weights(comparison):
    """비중이 각 분기 전체 평가금액 대비로 계산되는지 확인합니다."""
    apple = row_of(comparison, "037833100")

    assert apple["previous_weight"] == pytest.approx(1000 / PREVIOUS_TOTAL * 100)
    assert apple["current_weight"] == pytest.approx(1500 / CURRENT_TOTAL * 100)
    assert apple["weight_change_pct_point"] == pytest.approx(
        1500 / CURRENT_TOTAL * 100 - 1000 / PREVIOUS_TOTAL * 100
    )


def test_weights_add_up_to_100_percent(comparison):
    """각 분기 비중의 합계는 100%가 되어야 합니다."""
    assert comparison["previous_weight"].sum() == pytest.approx(100.0)
    assert comparison["current_weight"].sum() == pytest.approx(100.0)


def test_new_position_weight_change_is_positive(comparison):
    """신규 편입 종목의 이전 비중은 0, 비중 변화는 현재 비중과 같습니다."""
    occidental = row_of(comparison, "674599105")

    assert occidental["previous_weight"] == 0
    assert occidental["current_weight"] == pytest.approx(700 / CURRENT_TOTAL * 100)
    assert occidental["weight_change_pct_point"] == pytest.approx(
        occidental["current_weight"]
    )


def test_exited_position_weight_change_is_negative(comparison):
    """전량 매도 종목의 비중 변화는 이전 비중만큼 마이너스입니다."""
    bank_of_america = row_of(comparison, "060505104")

    assert bank_of_america["previous_weight"] == pytest.approx(
        300 / PREVIOUS_TOTAL * 100
    )
    assert bank_of_america["weight_change_pct_point"] == pytest.approx(
        -300 / PREVIOUS_TOTAL * 100
    )


# ---------------------------------------------------------------------------
# 7. 전체 평가금액 변화 계산 (요약 함수)
# ---------------------------------------------------------------------------


def test_summary_calculates_total_value_change(comparison):
    """전체 평가금액과 증감, 증감률이 맞는지 확인합니다."""
    summary = summarize_comparison(comparison)

    assert summary["previous_total_value"] == pytest.approx(PREVIOUS_TOTAL)
    assert summary["current_total_value"] == pytest.approx(CURRENT_TOTAL)
    assert summary["total_value_change"] == pytest.approx(CURRENT_TOTAL - PREVIOUS_TOTAL)
    assert summary["total_value_change_pct"] == pytest.approx(800 / 2100 * 100)


def test_summary_counts_each_change_status(comparison):
    """변화 구분별 종목 수가 맞는지 확인합니다."""
    summary = summarize_comparison(comparison)

    assert summary["new_position_count"] == 1  # OCCIDENTAL
    assert summary["exited_position_count"] == 1  # BANK OF AMERICA
    assert summary["increased_position_count"] == 1  # APPLE
    assert summary["decreased_position_count"] == 1  # COCA COLA
    assert summary["unchanged_position_count"] == 1  # WELLS FARGO


def test_summary_returns_expected_keys(comparison):
    """요약 딕셔너리에 필요한 9개 키가 모두 담겨 있는지 확인합니다."""
    summary = summarize_comparison(comparison)

    assert set(summary) == {
        "current_total_value",
        "previous_total_value",
        "total_value_change",
        "total_value_change_pct",
        "new_position_count",
        "exited_position_count",
        "increased_position_count",
        "decreased_position_count",
        "unchanged_position_count",
    }


# ---------------------------------------------------------------------------
# 8. 빈 입력 처리
# ---------------------------------------------------------------------------


def test_compare_holdings_handles_two_empty_inputs():
    """양쪽 입력이 비어 있어도 예외 없이 빈 표를 돌려줍니다."""
    result = compare_holdings([], [])

    assert result.empty
    assert list(result.columns) == COMPARISON_COLUMNS


def test_compare_holdings_handles_empty_previous_quarter():
    """이전 분기 데이터가 없으면 모든 종목이 '신규 편입'이 됩니다."""
    result = compare_holdings([], CURRENT_HOLDINGS)

    assert len(result) == 4
    assert set(result["change_status"]) == {STATUS_NEW}
    assert result["previous_weight"].sum() == 0
    # 이전 금액이 0이므로 변화율은 계산할 수 없습니다(0으로 나누기 방지).
    assert result["value_change_pct"].isna().all()


def test_compare_holdings_handles_empty_current_quarter():
    """현재 분기 데이터가 없으면 모든 종목이 '전량 매도'가 됩니다."""
    result = compare_holdings(PREVIOUS_HOLDINGS, [])

    assert len(result) == 4
    assert set(result["change_status"]) == {STATUS_EXITED}
    assert result["current_weight"].sum() == 0


def test_summarize_comparison_handles_empty_input():
    """빈 표를 요약해도 예외 없이 0으로 채운 결과를 돌려줍니다."""
    summary = summarize_comparison(compare_holdings([], []))

    assert summary["current_total_value"] == 0
    assert summary["previous_total_value"] == 0
    assert summary["total_value_change"] == 0
    # 이전 금액이 0이라 증감률은 계산할 수 없습니다.
    assert summary["total_value_change_pct"] is None
    assert summary["new_position_count"] == 0
    assert summary["exited_position_count"] == 0


def test_aggregate_holdings_handles_empty_input():
    """보유 종목이 없어도 열만 있는 빈 표를 돌려줍니다."""
    aggregated = aggregate_holdings([])

    assert aggregated.empty
    assert list(aggregated.columns) == [
        "cusip",
        "issuer_name",
        "class_title",
        "value_thousands",
        "shares",
    ]


# ---------------------------------------------------------------------------
# 9. 신규 편입 종목의 0으로 나누기 방지
# ---------------------------------------------------------------------------


def test_new_position_value_change_pct_is_not_a_number(comparison):
    """이전 금액이 0인 종목의 변화율은 무한대가 아니라 NaN이어야 합니다."""
    occidental = row_of(comparison, "674599105")

    assert pd.isna(occidental["value_change_pct"])
    # 0으로 나눈 결과(inf)가 섞여 들어가지 않았는지 전체 열을 확인합니다.
    assert not any(comparison["value_change_pct"].abs() == float("inf"))


def test_value_change_pct_is_none_friendly_for_display(comparison):
    """이전 금액이 있는 종목은 정상적으로 숫자 변화율이 나옵니다."""
    values = comparison.set_index("cusip")["value_change_pct"]

    assert values["037833100"] == pytest.approx(50.0)  # 1000 -> 1500
    assert pd.isna(values["674599105"])  # 신규 편입


# ---------------------------------------------------------------------------
# 10. 원본 입력 데이터 보존
# ---------------------------------------------------------------------------


def test_compare_holdings_does_not_modify_input_lists():
    """계산 과정에서 원본 입력이 바뀌지 않는지 확인합니다."""
    previous_backup = copy.deepcopy(PREVIOUS_HOLDINGS)
    current_backup = copy.deepcopy(CURRENT_HOLDINGS)

    compare_holdings(PREVIOUS_HOLDINGS, CURRENT_HOLDINGS)
    aggregate_holdings(PREVIOUS_HOLDINGS)

    assert PREVIOUS_HOLDINGS == previous_backup
    assert CURRENT_HOLDINGS == current_backup


def test_compare_holdings_does_not_modify_input_dataframe():
    """DataFrame을 넘겨받아도 원본 표가 바뀌지 않는지 확인합니다."""
    frame = pd.DataFrame(CURRENT_HOLDINGS)
    frame_backup = frame.copy(deep=True)

    compare_holdings(PREVIOUS_HOLDINGS, frame)

    pd.testing.assert_frame_equal(frame, frame_backup)


# ---------------------------------------------------------------------------
# 그 밖의 안전장치
# ---------------------------------------------------------------------------


def test_result_columns_and_order(comparison):
    """반환 표의 열 이름과 순서가 정해진 대로인지 확인합니다."""
    assert list(comparison.columns) == COMPARISON_COLUMNS


def test_result_is_sorted_by_current_value_descending(comparison):
    """현재 분기 평가금액이 큰 종목부터 정렬되는지 확인합니다."""
    values = comparison["current_value_thousands"].tolist()

    assert values == sorted(values, reverse=True)
    assert values == [1500, 700, 400, 300, 0]
    # 전량 매도 종목(현재 0)은 마지막에 놓입니다.
    assert comparison.iloc[-1]["change_status"] == STATUS_EXITED


def test_non_numeric_values_are_treated_as_zero():
    """숫자로 바꿀 수 없는 값은 0으로 처리되어 계산이 멈추지 않습니다."""
    messy_holdings = [
        {
            "issuer_name": "MESSY DATA CO",
            "class_title": "COM",
            "cusip": "222222222",
            "value_thousands": "알 수 없음",
            "shares": None,
        },
        {
            "issuer_name": "NORMAL CO",
            "class_title": "COM",
            "cusip": "333333333",
            "value_thousands": 100,
            "shares": 10,
        },
    ]

    result = compare_holdings(messy_holdings, messy_holdings)

    messy = row_of(result, "222222222")
    assert messy["current_value_thousands"] == 0
    assert messy["current_shares"] == 0
    assert messy["change_status"] == STATUS_UNCHANGED

    # 정상 종목의 비중은 나머지 금액(100)을 기준으로 100%가 됩니다.
    normal = row_of(result, "333333333")
    assert normal["current_weight"] == pytest.approx(100.0)


def test_missing_optional_columns_do_not_break_comparison():
    """일부 키가 아예 없는 데이터도 오류 없이 처리됩니다."""
    holdings_without_class_title = [
        {"cusip": "444444444", "issuer_name": "NO CLASS CO", "value_thousands": 50, "shares": 5},
    ]

    result = compare_holdings([], holdings_without_class_title)

    assert len(result) == 1
    assert result.iloc[0]["class_title"] == ""
    assert result.iloc[0]["change_status"] == STATUS_NEW


def test_value_increase_without_share_change_is_unchanged():
    """주가만 올라 평가금액이 늘고 보유수량은 같으면 '유지'로 봅니다."""
    previous = [
        {"cusip": "555555555", "issuer_name": "HOLD CO", "class_title": "COM", "value_thousands": 100, "shares": 10},
    ]
    current = [
        {"cusip": "555555555", "issuer_name": "HOLD CO", "class_title": "COM", "value_thousands": 150, "shares": 10},
    ]

    result = compare_holdings(previous, current)

    assert result.iloc[0]["change_status"] == STATUS_UNCHANGED
    assert result.iloc[0]["value_change_thousands"] == 50
    assert result.iloc[0]["shares_change"] == 0
