"""services/portfolio_analysis.py 테스트.

외부 API를 호출하지 않는 순수 계산 모듈이므로 네트워크 요청이나 mock이 필요 없습니다.
예시 보유 종목 목록을 직접 만들어 계산 결과를 검증합니다.
"""

import copy

import pandas as pd
import pytest

from services.portfolio_analysis import (
    AGGREGATED_COLUMNS,
    COMPARISON_COLUMNS,
    POSITION_KEY,
    PUT_CALL_EQUITY,
    SHARE_TYPE_UNKNOWN,
    STATUS_DECREASED,
    STATUS_EXITED,
    STATUS_INCREASED,
    STATUS_NEW,
    STATUS_UNCHANGED,
    aggregate_holdings,
    build_position_key,
    compare_holdings,
    is_fallback_position_key,
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
        "reported_value": 1000,
        "shares": 100,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 500,
        "shares": 50,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "BANK OF AMERICA CORP",
        "class_title": "COM",
        "cusip": "060505104",
        "reported_value": 300,
        "shares": 30,
        "share_type": "SH",
        "put_call": "",
    },
    # 같은 CUSIP이 두 줄로 나뉘어 공시된 경우(합산 대상).
    {
        "issuer_name": "WELLS FARGO & CO",
        "class_title": "COM",
        "cusip": "949746101",
        "reported_value": 100,
        "shares": 10,
        "share_type": "SH",
        "put_call": "",
    },
    {
        # 발행사명과 증권 종류가 비어 있어도, 다른 줄의 값을 쓰면 됩니다.
        "issuer_name": "",
        "class_title": "",
        "cusip": "949746101",
        "reported_value": 200,
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
        "reported_value": 1500,
        "shares": 120,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 400,
        "shares": 40,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "WELLS FARGO & CO",
        "class_title": "COM",
        "cusip": "949746101",
        "reported_value": 300,
        "shares": 30,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "OCCIDENTAL PETROLEUM CORP",
        "class_title": "COM",
        "cusip": "674599105",
        "reported_value": 700,
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


def row_by_key(table: pd.DataFrame, position_key: str) -> pd.Series:
    """포지션 키로 한 줄을 찾아 돌려주는 테스트 보조 함수."""
    matched = table[table[POSITION_KEY] == position_key]
    assert len(matched) == 1, f"포지션 {position_key} 줄이 정확히 1개여야 합니다."
    return matched.iloc[0]


def holding(
    cusip: str,
    issuer_name: str = "TEST CO",
    class_title: str = "COM",
    reported_value: int = 100,
    shares: int = 10,
    share_type: str = "SH",
    put_call: str = "",
) -> dict:
    """테스트용 보유 종목 한 줄을 만드는 보조 함수."""
    return {
        "issuer_name": issuer_name,
        "class_title": class_title,
        "cusip": cusip,
        "reported_value": reported_value,
        "shares": shares,
        "share_type": share_type,
        "put_call": put_call,
    }


# ---------------------------------------------------------------------------
# 1. 신규 편입 종목 분류
# ---------------------------------------------------------------------------


def test_classifies_new_position(comparison):
    """이전 분기에 없던 종목은 '신규 편입'으로 분류됩니다."""
    occidental = row_of(comparison, "674599105")

    assert occidental["change_status"] == STATUS_NEW
    assert occidental["issuer_name"] == "OCCIDENTAL PETROLEUM CORP"
    assert occidental["previous_reported_value"] == 0
    assert occidental["previous_shares"] == 0
    assert occidental["current_reported_value"] == 700
    assert occidental["current_shares"] == 70
    assert occidental["reported_value_change"] == 700
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
    assert bank_of_america["previous_reported_value"] == 300
    assert bank_of_america["current_reported_value"] == 0
    assert bank_of_america["current_shares"] == 0
    assert bank_of_america["reported_value_change"] == -300
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
    assert apple["reported_value_change"] == 500
    assert apple["value_change_pct"] == pytest.approx(50.0)


def test_classifies_decreased_position(comparison):
    """보유수량이 줄어든 종목은 '보유 축소'로 분류됩니다."""
    coca_cola = row_of(comparison, "191216100")

    assert coca_cola["change_status"] == STATUS_DECREASED
    assert coca_cola["previous_shares"] == 50
    assert coca_cola["current_shares"] == 40
    assert coca_cola["shares_change"] == -10
    assert coca_cola["reported_value_change"] == -100
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
    assert wells_fargo["reported_value"] == 300  # 100 + 200
    assert wells_fargo["shares"] == 30  # 10 + 20
    # 비어 있는 값 대신, 비어 있지 않은 첫 번째 값을 씁니다.
    assert wells_fargo["issuer_name"] == "WELLS FARGO & CO"
    assert wells_fargo["class_title"] == "COM"


def test_compare_holdings_uses_summed_duplicate_rows(comparison):
    """비교 결과에도 합산된 값이 반영되는지 확인합니다."""
    wells_fargo = row_of(comparison, "949746101")

    assert wells_fargo["previous_reported_value"] == 300
    assert wells_fargo["previous_shares"] == 30


def test_aggregate_holdings_uses_first_valid_issuer_name():
    """앞쪽 줄의 이름이 비어 있으면 뒤쪽 줄의 이름을 사용합니다."""
    holdings = [
        {"cusip": "111111111", "issuer_name": "", "class_title": "", "reported_value": 10, "shares": 1},
        {"cusip": "111111111", "issuer_name": "SECOND ROW CO", "class_title": "COM", "reported_value": 20, "shares": 2},
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["issuer_name"] == "SECOND ROW CO"
    assert aggregated.iloc[0]["class_title"] == "COM"
    assert aggregated.iloc[0]["reported_value"] == 30


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
    assert list(aggregated.columns) == AGGREGATED_COLUMNS
    assert list(aggregated.columns) == [
        POSITION_KEY,
        "cusip",
        "issuer_name",
        "class_title",
        "put_call",
        "share_type",
        "reported_value",
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
    values = comparison["current_reported_value"].tolist()

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
            "reported_value": "알 수 없음",
            "shares": None,
        },
        {
            "issuer_name": "NORMAL CO",
            "class_title": "COM",
            "cusip": "333333333",
            "reported_value": 100,
            "shares": 10,
        },
    ]

    result = compare_holdings(messy_holdings, messy_holdings)

    messy = row_of(result, "222222222")
    assert messy["current_reported_value"] == 0
    assert messy["current_shares"] == 0
    assert messy["change_status"] == STATUS_UNCHANGED

    # 정상 종목의 비중은 나머지 금액(100)을 기준으로 100%가 됩니다.
    normal = row_of(result, "333333333")
    assert normal["current_weight"] == pytest.approx(100.0)


def test_missing_optional_columns_do_not_break_comparison():
    """일부 키가 아예 없는 데이터도 오류 없이 처리됩니다."""
    holdings_without_class_title = [
        {"cusip": "444444444", "issuer_name": "NO CLASS CO", "reported_value": 50, "shares": 5},
    ]

    result = compare_holdings([], holdings_without_class_title)

    assert len(result) == 1
    assert result.iloc[0]["class_title"] == ""
    assert result.iloc[0]["change_status"] == STATUS_NEW


def test_value_increase_without_share_change_is_unchanged():
    """주가만 올라 평가금액이 늘고 보유수량은 같으면 '유지'로 봅니다."""
    previous = [
        {"cusip": "555555555", "issuer_name": "HOLD CO", "class_title": "COM", "reported_value": 100, "shares": 10},
    ]
    current = [
        {"cusip": "555555555", "issuer_name": "HOLD CO", "class_title": "COM", "reported_value": 150, "shares": 10},
    ]

    result = compare_holdings(previous, current)

    assert result.iloc[0]["change_status"] == STATUS_UNCHANGED
    assert result.iloc[0]["reported_value_change"] == 50
    assert result.iloc[0]["shares_change"] == 0


# ---------------------------------------------------------------------------
# 11. 포지션 키 생성 규칙
# ---------------------------------------------------------------------------


def test_build_position_key_uses_cusip_put_call_and_share_type():
    """포지션 키는 CUSIP·옵션 구분·수량 단위를 이어 붙인 값입니다."""
    assert build_position_key("037833100", "", "SH") == "037833100|EQUITY|SH"
    assert build_position_key("037833100", "Call", "SH") == "037833100|CALL|SH"
    assert build_position_key("037833100", "Put", "SH") == "037833100|PUT|SH"


def test_build_position_key_normalizes_spaces_and_case():
    """앞뒤 공백을 없애고 대문자로 통일하므로 표기 차이는 같은 포지션이 됩니다."""
    assert build_position_key(" 037833100 ", " call ", " sh ") == "037833100|CALL|SH"


def test_build_position_key_fills_missing_put_call_and_share_type():
    """put_call이 비면 EQUITY, share_type이 비면 UNKNOWN으로 채웁니다."""
    key = build_position_key("037833100")

    assert key == f"037833100|{PUT_CALL_EQUITY}|{SHARE_TYPE_UNKNOWN}"
    assert not is_fallback_position_key(key)


def test_build_position_key_falls_back_to_issuer_name_without_cusip():
    """CUSIP이 없으면 발행사명·증권 종류로 대체 키를 만들고 표시를 남깁니다."""
    key = build_position_key(
        "", put_call="", share_type="SH", issuer_name="Mystery Co", class_title="com"
    )

    assert is_fallback_position_key(key)
    assert "MYSTERY CO" in key
    # 발행사명이 다르면 대체 키도 달라야 합니다.
    other = build_position_key(
        "", put_call="", share_type="SH", issuer_name="Other Co", class_title="com"
    )
    assert key != other


# ---------------------------------------------------------------------------
# 12. 포지션 키 기준 중복 행 합산
# ---------------------------------------------------------------------------


def test_same_position_duplicate_rows_are_summed():
    """CUSIP·put_call·share_type이 모두 같은 중복 행은 금액과 수량을 더합니다."""
    holdings = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
        holding("037833100", "APPLE INC", reported_value=250, shares=25),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row[POSITION_KEY] == "037833100|EQUITY|SH"
    assert row["reported_value"] == 1250
    assert row["shares"] == 125


def test_option_duplicate_rows_are_summed_within_same_option_type():
    """같은 Call 옵션이 여러 줄로 나뉘어도 하나로 합산됩니다."""
    holdings = [
        holding("037833100", "APPLE INC", reported_value=100, shares=10, put_call="Call"),
        holding("037833100", "APPLE INC", reported_value=300, shares=30, put_call="CALL"),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["reported_value"] == 400
    assert aggregated.iloc[0]["shares"] == 40


def test_equity_and_call_positions_are_kept_separate():
    """같은 CUSIP이라도 일반 주식과 CALL은 별도 포지션으로 남습니다."""
    holdings = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
        holding("037833100", "APPLE INC", reported_value=200, shares=20, put_call="CALL"),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 2

    equity = row_by_key(aggregated, "037833100|EQUITY|SH")
    call = row_by_key(aggregated, "037833100|CALL|SH")

    assert equity["reported_value"] == 1000
    assert equity["shares"] == 100
    assert call["reported_value"] == 200
    assert call["shares"] == 20
    # 대표값으로 남는 put_call은 공시 원문 표기를 그대로 유지합니다.
    assert equity["put_call"] == ""
    assert call["put_call"] == "CALL"


def test_put_and_call_positions_are_kept_separate():
    """같은 CUSIP이라도 PUT과 CALL은 별도 포지션으로 남습니다."""
    holdings = [
        holding("037833100", "APPLE INC", reported_value=700, shares=70, put_call="PUT"),
        holding("037833100", "APPLE INC", reported_value=400, shares=40, put_call="CALL"),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 2
    assert row_by_key(aggregated, "037833100|PUT|SH")["reported_value"] == 700
    assert row_by_key(aggregated, "037833100|CALL|SH")["reported_value"] == 400


def test_different_share_type_keeps_positions_separate():
    """CUSIP과 put_call이 같아도 수량 단위(share_type)가 다르면 별도 포지션입니다."""
    holdings = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100, share_type="SH"),
        holding("037833100", "APPLE INC", reported_value=500, shares=50, share_type="PRN"),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 2
    assert row_by_key(aggregated, "037833100|EQUITY|SH")["shares"] == 100
    assert row_by_key(aggregated, "037833100|EQUITY|PRN")["shares"] == 50


def test_rows_without_cusip_are_not_merged_together():
    """CUSIP이 없는 서로 다른 종목이 하나로 합쳐지지 않습니다."""
    holdings = [
        holding("", "FIRST MYSTERY CO", reported_value=100, shares=10),
        holding("", "SECOND MYSTERY CO", reported_value=200, shares=20),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 2
    assert all(is_fallback_position_key(key) for key in aggregated[POSITION_KEY])
    assert set(aggregated["issuer_name"]) == {"FIRST MYSTERY CO", "SECOND MYSTERY CO"}
    assert sorted(aggregated["reported_value"]) == [100, 200]


def test_rows_without_cusip_but_same_issuer_are_summed():
    """CUSIP이 없어도 발행사명·증권 종류·옵션 구분이 같으면 같은 포지션입니다."""
    holdings = [
        holding("", "SAME MYSTERY CO", reported_value=100, shares=10),
        holding("", "SAME MYSTERY CO", reported_value=150, shares=15),
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["reported_value"] == 250
    assert aggregated.iloc[0]["shares"] == 25


def test_aggregate_holdings_keeps_first_valid_representative_values():
    """대표값으로 빈 문자열이 아니라 비어 있지 않은 첫 값을 씁니다."""
    holdings = [
        {
            "issuer_name": "",
            "class_title": "",
            "cusip": "037833100",
            "reported_value": 100,
            "shares": 10,
            "share_type": "SH",
            "put_call": "CALL",
        },
        {
            "issuer_name": "APPLE INC",
            "class_title": "COM",
            "cusip": "037833100",
            "reported_value": 200,
            "shares": 20,
            "share_type": "SH",
            "put_call": "CALL",
        },
    ]

    aggregated = aggregate_holdings(holdings)

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["issuer_name"] == "APPLE INC"
    assert row["class_title"] == "COM"
    assert row["cusip"] == "037833100"
    assert row["put_call"] == "CALL"
    assert row["share_type"] == "SH"


def test_aggregate_holdings_does_not_modify_input_option_rows():
    """옵션 행이 섞인 입력도 계산 과정에서 원본이 바뀌지 않습니다."""
    holdings = [
        holding("037833100", "APPLE INC", put_call="CALL"),
        holding("037833100", "APPLE INC", put_call=""),
        holding("", "NO CUSIP CO"),
    ]
    backup = copy.deepcopy(holdings)

    aggregate_holdings(holdings)
    compare_holdings(holdings, holdings)

    assert holdings == backup


# ---------------------------------------------------------------------------
# 13. 포지션 키 기준 두 분기 결합
# ---------------------------------------------------------------------------


def test_comparison_includes_position_key():
    """비교 결과에 포지션 키가 함께 담깁니다."""
    result = compare_holdings(PREVIOUS_HOLDINGS, CURRENT_HOLDINGS)

    assert POSITION_KEY in result.columns
    assert row_of(result, "037833100")[POSITION_KEY] == "037833100|EQUITY|SH"
    # 포지션 키는 결과 표에서 중복되지 않습니다.
    assert not result[POSITION_KEY].duplicated().any()


def test_comparison_keeps_put_call_and_share_type_columns():
    """화면에서 쓰는 put_call, share_type 정보가 비교 결과에 남아 있습니다."""
    result = compare_holdings(PREVIOUS_HOLDINGS, CURRENT_HOLDINGS)

    apple = row_of(result, "037833100")
    assert apple["share_type"] == "SH"
    assert apple["put_call"] == ""


def test_comparison_matches_equity_and_option_positions_separately():
    """일반 주식과 옵션은 각각 짝을 맞춰 비교합니다(서로 섞이지 않습니다)."""
    previous = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
        holding("037833100", "APPLE INC", reported_value=300, shares=30, put_call="CALL"),
    ]
    current = [
        holding("037833100", "APPLE INC", reported_value=900, shares=80),
        holding("037833100", "APPLE INC", reported_value=500, shares=50, put_call="CALL"),
    ]

    result = compare_holdings(previous, current)

    assert len(result) == 2

    equity = row_by_key(result, "037833100|EQUITY|SH")
    call = row_by_key(result, "037833100|CALL|SH")

    assert equity["previous_shares"] == 100
    assert equity["current_shares"] == 80
    assert equity["change_status"] == STATUS_DECREASED

    assert call["previous_shares"] == 30
    assert call["current_shares"] == 50
    assert call["change_status"] == STATUS_INCREASED


def test_position_only_in_previous_quarter_is_exited():
    """이전 분기에만 있는 옵션 포지션은 '전량 매도'로 분류됩니다."""
    previous = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
        holding("037833100", "APPLE INC", reported_value=300, shares=30, put_call="PUT"),
    ]
    current = [
        holding("037833100", "APPLE INC", reported_value=1100, shares=100),
    ]

    result = compare_holdings(previous, current)

    put_position = row_by_key(result, "037833100|PUT|SH")
    assert put_position["change_status"] == STATUS_EXITED
    assert put_position["previous_shares"] == 30
    assert put_position["current_shares"] == 0
    assert put_position["current_reported_value"] == 0

    # 보통주 보유는 옵션 청산과 무관하게 '유지'로 남습니다.
    assert row_by_key(result, "037833100|EQUITY|SH")["change_status"] == STATUS_UNCHANGED


def test_position_only_in_current_quarter_is_new():
    """현재 분기에만 있는 옵션 포지션은 '신규 편입'으로 분류됩니다."""
    previous = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
    ]
    current = [
        holding("037833100", "APPLE INC", reported_value=1000, shares=100),
        holding("037833100", "APPLE INC", reported_value=300, shares=30, put_call="CALL"),
    ]

    result = compare_holdings(previous, current)

    call_position = row_by_key(result, "037833100|CALL|SH")
    assert call_position["change_status"] == STATUS_NEW
    assert call_position["previous_shares"] == 0
    assert call_position["previous_reported_value"] == 0
    assert call_position["current_shares"] == 30


def test_comparison_does_not_pair_different_share_types():
    """수량 단위가 바뀌면 같은 CUSIP이어도 별개 포지션으로 비교합니다."""
    previous = [holding("037833100", "APPLE INC", shares=100, share_type="PRN")]
    current = [holding("037833100", "APPLE INC", shares=100, share_type="SH")]

    result = compare_holdings(previous, current)

    assert len(result) == 2
    assert row_by_key(result, "037833100|EQUITY|PRN")["change_status"] == STATUS_EXITED
    assert row_by_key(result, "037833100|EQUITY|SH")["change_status"] == STATUS_NEW


def test_comparison_does_not_merge_rows_without_cusip():
    """CUSIP이 없는 서로 다른 종목은 비교 단계에서도 합쳐지지 않습니다."""
    previous = [
        holding("", "FIRST MYSTERY CO", reported_value=100, shares=10),
        holding("", "SECOND MYSTERY CO", reported_value=200, shares=20),
    ]
    current = [
        holding("", "FIRST MYSTERY CO", reported_value=150, shares=15),
    ]

    result = compare_holdings(previous, current)

    assert len(result) == 2

    first = result[result["issuer_name"] == "FIRST MYSTERY CO"].iloc[0]
    second = result[result["issuer_name"] == "SECOND MYSTERY CO"].iloc[0]

    assert first["change_status"] == STATUS_INCREASED
    assert first["current_shares"] == 15
    assert second["change_status"] == STATUS_EXITED
    assert second["current_shares"] == 0
