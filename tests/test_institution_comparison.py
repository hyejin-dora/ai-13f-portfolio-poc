"""services/institution_comparison.py 테스트.

외부 API를 호출하지 않는 순수 계산 모듈이므로 네트워크 요청이나 mock이 필요 없습니다.
예시 공시 목록과 보유 종목 목록을 직접 만들어 계산 결과를 검증합니다.
"""

import copy

import pandas as pd
import pytest

from services.institution_comparison import (
    HOLDING_TYPE,
    HOLDING_TYPE_COMMON,
    HOLDING_TYPE_LEFT_ONLY,
    HOLDING_TYPE_RIGHT_ONLY,
    INSTITUTION_COMPARISON_COLUMNS,
    PORTFOLIO_COLUMNS,
    compare_institution_portfolios,
    find_common_report_dates,
    index_filings_by_report_date,
    prepare_institution_portfolio,
    summarize_institution_comparison,
)
from services.portfolio_analysis import POSITION_KEY, build_position_key

# ---------------------------------------------------------------------------
# 예시 데이터 만들기 도우미
# ---------------------------------------------------------------------------


def filing(report_date, filing_date, accession_number="0000000000-00-000000"):
    """공시 한 건을 나타내는 딕셔너리를 만듭니다(sec_client 결과와 같은 구조)."""
    return {
        "accession_number": accession_number,
        "filing_date": filing_date,
        "report_date": report_date,
        "primary_document": "primary_doc.xml",
    }


def holding(
    cusip,
    issuer_name,
    reported_value=1000,
    shares=100,
    share_type="SH",
    put_call="",
    class_title="COM",
):
    """보유 종목 한 줄을 나타내는 딕셔너리를 만듭니다."""
    return {
        "issuer_name": issuer_name,
        "class_title": class_title,
        "cusip": cusip,
        "reported_value": reported_value,
        "shares": shares,
        "share_type": share_type,
        "put_call": put_call,
    }


def row_for(comparison, cusip, put_call="", share_type="SH"):
    """비교 결과에서 특정 포지션 한 줄을 꺼냅니다."""
    key = build_position_key(cusip=cusip, put_call=put_call, share_type=share_type)
    matched = comparison[comparison[POSITION_KEY] == key]
    assert len(matched) == 1, f"포지션 {key}가 정확히 한 줄이어야 합니다."
    return matched.iloc[0]


# ---------------------------------------------------------------------------
# 예시 포트폴리오
#
# 기관 A 합계: 6000 + 3000 + 1000 = 10000
#   APPLE       6000 -> 60%
#   COCA COLA   3000 -> 30%
#   AMEX        1000 -> 10%
#
# 기관 B 합계: 1000 + 2000 + 1000 = 4000
#   APPLE       1000 -> 25%
#   COCA COLA   2000 -> 50%
#   MICROSOFT   1000 -> 25%
#
# 공통 보유: APPLE, COCA COLA / 기관 A 단독: AMEX / 기관 B 단독: MICROSOFT
# ---------------------------------------------------------------------------

APPLE = "037833100"
COCA_COLA = "191216100"
AMEX = "025816109"
MICROSOFT = "594918104"

LEFT_HOLDINGS = [
    holding(APPLE, "APPLE INC", reported_value=6000, shares=600),
    holding(COCA_COLA, "COCA COLA CO", reported_value=3000, shares=300),
    holding(AMEX, "AMERICAN EXPRESS CO", reported_value=1000, shares=100),
]

RIGHT_HOLDINGS = [
    holding(APPLE, "APPLE INC", reported_value=1000, shares=100),
    holding(COCA_COLA, "COCA COLA CO", reported_value=2000, shares=200),
    holding(MICROSOFT, "MICROSOFT CORP", reported_value=1000, shares=50),
]


@pytest.fixture
def comparison():
    """예시 두 기관의 비교 결과 표."""
    return compare_institution_portfolios(LEFT_HOLDINGS, RIGHT_HOLDINGS)


# ---------------------------------------------------------------------------
# 공시 기준일 매핑 (index_filings_by_report_date)
# ---------------------------------------------------------------------------


def test_index_filings_uses_report_date_as_key():
    """공시 목록이 기준일을 열쇠로 하는 딕셔너리가 됩니다."""
    filings = [
        filing("2025-06-30", "2025-08-14"),
        filing("2025-03-31", "2025-05-15"),
    ]

    indexed = index_filings_by_report_date(filings)

    assert set(indexed) == {"2025-06-30", "2025-03-31"}
    assert indexed["2025-06-30"]["filing_date"] == "2025-08-14"


def test_index_filings_skips_missing_report_date():
    """기준일이 없거나 비어 있는 공시는 제외합니다."""
    filings = [
        filing("2025-06-30", "2025-08-14"),
        filing("", "2025-08-15"),
        filing(None, "2025-08-16"),
        {"accession_number": "0000000000-00-000001", "filing_date": "2025-08-17"},
    ]

    indexed = index_filings_by_report_date(filings)

    assert list(indexed) == ["2025-06-30"]


def test_index_filings_keeps_latest_filing_date_for_same_report_date():
    """같은 기준일 공시가 여러 건이면 제출일이 가장 늦은 것을 고릅니다."""
    filings = [
        filing("2025-06-30", "2025-08-20", accession_number="LATEST"),
        filing("2025-06-30", "2025-08-14", accession_number="OLDEST"),
    ]

    indexed = index_filings_by_report_date(filings)

    assert indexed["2025-06-30"]["accession_number"] == "LATEST"


def test_index_filings_prefers_last_entry_when_filing_date_ties():
    """제출일까지 같으면 목록에서 나중에 나온 공시를 씁니다."""
    filings = [
        filing("2025-06-30", "2025-08-14", accession_number="FIRST"),
        filing("2025-06-30", "2025-08-14", accession_number="SECOND"),
    ]

    indexed = index_filings_by_report_date(filings)

    assert indexed["2025-06-30"]["accession_number"] == "SECOND"


def test_index_filings_handles_missing_filing_date():
    """제출일이 없는 공시가 섞여 있어도 제출일이 있는 공시를 고릅니다."""
    filings = [
        filing("2025-06-30", "2025-08-14", accession_number="WITH_DATE"),
        filing("2025-06-30", None, accession_number="NO_DATE"),
    ]

    indexed = index_filings_by_report_date(filings)

    assert indexed["2025-06-30"]["accession_number"] == "WITH_DATE"


def test_index_filings_handles_empty_input():
    """빈 목록이나 None을 넘겨도 빈 딕셔너리를 돌려줍니다."""
    assert index_filings_by_report_date([]) == {}
    assert index_filings_by_report_date(None) == {}


def test_index_filings_does_not_modify_input():
    """정리 과정에서 원본 공시 목록이 바뀌지 않습니다."""
    filings = [
        filing("2025-06-30", "2025-08-14"),
        filing("2025-03-31", "2025-05-15"),
    ]
    backup = copy.deepcopy(filings)

    indexed = index_filings_by_report_date(filings)
    # 결과 딕셔너리를 고쳐도 원본에는 영향이 없어야 합니다.
    indexed["2025-06-30"]["filing_date"] = "바뀐 값"

    assert filings == backup


# ---------------------------------------------------------------------------
# 공통 분기 찾기 (find_common_report_dates)
# ---------------------------------------------------------------------------


def test_common_report_dates_are_sorted_newest_first():
    """두 기관 모두 공시한 분기를 최신순으로 돌려줍니다."""
    left = [
        filing("2025-03-31", "2025-05-15"),
        filing("2025-06-30", "2025-08-14"),
        filing("2024-12-31", "2025-02-14"),
    ]
    right = [
        filing("2024-12-31", "2025-02-10"),
        filing("2025-06-30", "2025-08-12"),
        filing("2025-03-31", "2025-05-13"),
    ]

    assert find_common_report_dates(left, right) == [
        "2025-06-30",
        "2025-03-31",
        "2024-12-31",
    ]


def test_common_report_dates_ignore_quarters_only_one_side_has():
    """한쪽에만 있는 분기는 공통 분기가 아닙니다."""
    left = [filing("2025-06-30", "2025-08-14"), filing("2025-03-31", "2025-05-15")]
    right = [filing("2025-06-30", "2025-08-12")]

    assert find_common_report_dates(left, right) == ["2025-06-30"]


def test_common_report_dates_use_report_date_not_filing_date():
    """제출일이 전혀 달라도 기준일이 같으면 공통 분기입니다."""
    left = [filing("2025-06-30", "2025-07-01")]
    right = [filing("2025-06-30", "2025-08-14")]

    assert find_common_report_dates(left, right) == ["2025-06-30"]


def test_common_report_dates_do_not_match_by_filing_date():
    """제출일이 같아도 기준일이 다르면 공통 분기가 아닙니다."""
    left = [filing("2025-06-30", "2025-08-14")]
    right = [filing("2025-03-31", "2025-08-14")]

    assert find_common_report_dates(left, right) == []


def test_common_report_dates_returns_empty_list_without_overlap():
    """겹치는 분기가 없으면 빈 목록을 돌려줍니다."""
    left = [filing("2025-06-30", "2025-08-14")]
    right = [filing("2024-12-31", "2025-02-14")]

    assert find_common_report_dates(left, right) == []


def test_common_report_dates_handles_empty_inputs():
    """공시 목록이 비어 있으면 빈 목록을 돌려줍니다."""
    assert find_common_report_dates([], []) == []
    assert find_common_report_dates([filing("2025-06-30", "2025-08-14")], []) == []


# ---------------------------------------------------------------------------
# 기관 포트폴리오 전처리 (prepare_institution_portfolio)
# ---------------------------------------------------------------------------


def test_prepare_portfolio_columns():
    """정리한 표가 정해진 열을 모두 가집니다."""
    portfolio = prepare_institution_portfolio(LEFT_HOLDINGS)

    assert list(portfolio.columns) == PORTFOLIO_COLUMNS


def test_prepare_portfolio_calculates_weights():
    """각 포지션 비중이 전체 평가금액 대비 비율(%)로 계산됩니다."""
    portfolio = prepare_institution_portfolio(LEFT_HOLDINGS)
    weights = dict(zip(portfolio["cusip"], portfolio["portfolio_weight_pct"]))

    assert weights[APPLE] == pytest.approx(60.0)
    assert weights[COCA_COLA] == pytest.approx(30.0)
    assert weights[AMEX] == pytest.approx(10.0)


def test_prepare_portfolio_weights_add_up_to_100_percent():
    """비중의 합은 100%입니다."""
    portfolio = prepare_institution_portfolio(LEFT_HOLDINGS)

    assert portfolio["portfolio_weight_pct"].sum() == pytest.approx(100.0)


def test_prepare_portfolio_sums_duplicate_position_rows():
    """같은 포지션이 여러 줄로 공시되어도 먼저 합산합니다."""
    holdings = [
        holding(APPLE, "APPLE INC", reported_value=600, shares=60),
        holding(APPLE, "APPLE INC", reported_value=400, shares=40),
    ]

    portfolio = prepare_institution_portfolio(holdings)

    assert len(portfolio) == 1
    assert portfolio.iloc[0]["reported_value"] == pytest.approx(1000.0)
    assert portfolio.iloc[0]["shares"] == pytest.approx(100.0)
    assert portfolio.iloc[0]["portfolio_weight_pct"] == pytest.approx(100.0)


def test_prepare_portfolio_treats_missing_numbers_as_zero():
    """평가금액과 보유수량의 결측치는 0으로 처리합니다."""
    holdings = [
        holding(APPLE, "APPLE INC", reported_value=1000, shares=100),
        holding(COCA_COLA, "COCA COLA CO", reported_value=None, shares=None),
    ]

    portfolio = prepare_institution_portfolio(holdings)
    coca_cola = portfolio[portfolio["cusip"] == COCA_COLA].iloc[0]

    assert coca_cola["reported_value"] == pytest.approx(0.0)
    assert coca_cola["shares"] == pytest.approx(0.0)
    assert coca_cola["portfolio_weight_pct"] == pytest.approx(0.0)


def test_prepare_portfolio_sets_all_weights_to_zero_when_total_is_zero():
    """전체 평가금액이 0이면 모든 비중을 0으로 둡니다."""
    holdings = [
        holding(APPLE, "APPLE INC", reported_value=0, shares=100),
        holding(COCA_COLA, "COCA COLA CO", reported_value=0, shares=50),
    ]

    portfolio = prepare_institution_portfolio(holdings)

    assert list(portfolio["portfolio_weight_pct"]) == [0.0, 0.0]


def test_prepare_portfolio_handles_empty_input():
    """보유 종목이 없으면 열만 있는 빈 표를 돌려줍니다."""
    portfolio = prepare_institution_portfolio([])

    assert portfolio.empty
    assert list(portfolio.columns) == PORTFOLIO_COLUMNS


def test_prepare_portfolio_does_not_modify_input():
    """전처리 과정에서 원본 보유 종목 목록이 바뀌지 않습니다."""
    backup = copy.deepcopy(LEFT_HOLDINGS)

    prepare_institution_portfolio(LEFT_HOLDINGS)

    assert LEFT_HOLDINGS == backup


# ---------------------------------------------------------------------------
# 두 기관 비교 (compare_institution_portfolios)
# ---------------------------------------------------------------------------


def test_comparison_columns(comparison):
    """비교 결과가 정해진 열을 모두 가집니다."""
    assert list(comparison.columns) == INSTITUTION_COMPARISON_COLUMNS


def test_comparison_includes_every_position(comparison):
    """양쪽 보유 종목을 합친 개수만큼 줄이 만들어집니다."""
    assert len(comparison) == 4


def test_classifies_common_holding(comparison):
    """두 기관이 함께 보유한 종목은 '공통 보유'입니다."""
    apple = row_for(comparison, APPLE)

    assert apple[HOLDING_TYPE] == HOLDING_TYPE_COMMON
    assert apple["left_reported_value"] == pytest.approx(6000.0)
    assert apple["right_reported_value"] == pytest.approx(1000.0)


def test_classifies_left_only_holding(comparison):
    """기관 A만 보유한 종목은 '기관 A 단독'이고 기관 B 값은 0입니다."""
    amex = row_for(comparison, AMEX)

    assert amex[HOLDING_TYPE] == HOLDING_TYPE_LEFT_ONLY
    assert amex["left_reported_value"] == pytest.approx(1000.0)
    assert amex["right_reported_value"] == pytest.approx(0.0)
    assert amex["right_shares"] == pytest.approx(0.0)
    assert amex["right_weight_pct"] == pytest.approx(0.0)


def test_classifies_right_only_holding(comparison):
    """기관 B만 보유한 종목은 '기관 B 단독'이고 기관 A 값은 0입니다."""
    microsoft = row_for(comparison, MICROSOFT)

    assert microsoft[HOLDING_TYPE] == HOLDING_TYPE_RIGHT_ONLY
    assert microsoft["left_reported_value"] == pytest.approx(0.0)
    assert microsoft["right_reported_value"] == pytest.approx(1000.0)


def test_comparison_keeps_issuer_name_for_right_only_position(comparison):
    """한쪽만 보유한 종목도 종목 정보가 비어 있지 않습니다."""
    microsoft = row_for(comparison, MICROSOFT)

    assert microsoft["issuer_name"] == "MICROSOFT CORP"
    assert microsoft["cusip"] == MICROSOFT
    assert microsoft["class_title"] == "COM"


def test_comparison_fills_issuer_name_from_the_other_side():
    """한쪽 공시에만 종목명이 적혀 있으면 그 값을 씁니다."""
    left = [holding(APPLE, "", reported_value=1000, shares=100)]
    right = [holding(APPLE, "APPLE INC", reported_value=1000, shares=100)]

    comparison = compare_institution_portfolios(left, right)

    assert row_for(comparison, APPLE)["issuer_name"] == "APPLE INC"


def test_comparison_calculates_weight_gap(comparison):
    """비중 차이는 기관 A 비중에서 기관 B 비중을 뺀 퍼센트포인트입니다."""
    apple = row_for(comparison, APPLE)
    coca_cola = row_for(comparison, COCA_COLA)

    # APPLE: 60% - 25% = +35%p (기관 A가 더 많이 담음)
    assert apple["weight_gap_pct_point"] == pytest.approx(35.0)
    # COCA COLA: 30% - 50% = -20%p (기관 B가 더 많이 담음)
    assert coca_cola["weight_gap_pct_point"] == pytest.approx(-20.0)


def test_weight_gap_equals_left_minus_right(comparison):
    """모든 줄에서 비중 차이가 두 비중의 차와 같습니다."""
    expected = comparison["left_weight_pct"] - comparison["right_weight_pct"]

    pd.testing.assert_series_equal(
        comparison["weight_gap_pct_point"], expected, check_names=False
    )


def test_comparison_sums_duplicate_position_rows_before_matching():
    """같은 포지션의 중복 행을 먼저 합산한 뒤 비교합니다."""
    left = [
        holding(APPLE, "APPLE INC", reported_value=600, shares=60),
        holding(APPLE, "APPLE INC", reported_value=400, shares=40),
    ]
    right = [holding(APPLE, "APPLE INC", reported_value=1000, shares=100)]

    comparison = compare_institution_portfolios(left, right)
    apple = row_for(comparison, APPLE)

    assert len(comparison) == 1
    assert apple["left_reported_value"] == pytest.approx(1000.0)
    assert apple["left_shares"] == pytest.approx(100.0)
    assert apple[HOLDING_TYPE] == HOLDING_TYPE_COMMON


def test_comparison_keeps_equity_call_and_put_separate():
    """CUSIP이 같아도 보통주·Call·Put은 서로 다른 포지션으로 남습니다."""
    left = [
        holding(APPLE, "APPLE INC", reported_value=1000, shares=100),
        holding(APPLE, "APPLE INC", reported_value=500, shares=50, put_call="CALL"),
        holding(APPLE, "APPLE INC", reported_value=300, shares=30, put_call="PUT"),
    ]
    right = [holding(APPLE, "APPLE INC", reported_value=1000, shares=100)]

    comparison = compare_institution_portfolios(left, right)

    assert len(comparison) == 3
    assert row_for(comparison, APPLE)[HOLDING_TYPE] == HOLDING_TYPE_COMMON
    assert (
        row_for(comparison, APPLE, put_call="CALL")[HOLDING_TYPE]
        == HOLDING_TYPE_LEFT_ONLY
    )
    assert (
        row_for(comparison, APPLE, put_call="PUT")[HOLDING_TYPE]
        == HOLDING_TYPE_LEFT_ONLY
    )


def test_comparison_handles_zero_total_reported_value():
    """전체 평가금액이 0이어도 비교가 되고 비중은 모두 0입니다."""
    left = [holding(APPLE, "APPLE INC", reported_value=0, shares=100)]
    right = [holding(APPLE, "APPLE INC", reported_value=0, shares=50)]

    comparison = compare_institution_portfolios(left, right)
    apple = row_for(comparison, APPLE)

    assert apple[HOLDING_TYPE] == HOLDING_TYPE_COMMON
    assert apple["left_weight_pct"] == pytest.approx(0.0)
    assert apple["right_weight_pct"] == pytest.approx(0.0)
    assert apple["weight_gap_pct_point"] == pytest.approx(0.0)
    assert apple["left_shares"] == pytest.approx(100.0)


def test_comparison_handles_two_empty_portfolios():
    """양쪽 모두 보유 종목이 없으면 열만 있는 빈 표를 돌려줍니다."""
    comparison = compare_institution_portfolios([], [])

    assert comparison.empty
    assert list(comparison.columns) == INSTITUTION_COMPARISON_COLUMNS


def test_comparison_handles_one_empty_portfolio():
    """한쪽만 비어 있으면 나머지 기관 단독 보유로 정리됩니다."""
    comparison = compare_institution_portfolios([], RIGHT_HOLDINGS)

    assert len(comparison) == 3
    assert set(comparison[HOLDING_TYPE]) == {HOLDING_TYPE_RIGHT_ONLY}


def test_comparison_does_not_modify_input_lists():
    """비교 과정에서 원본 목록이 바뀌지 않습니다."""
    left_backup = copy.deepcopy(LEFT_HOLDINGS)
    right_backup = copy.deepcopy(RIGHT_HOLDINGS)

    compare_institution_portfolios(LEFT_HOLDINGS, RIGHT_HOLDINGS)

    assert LEFT_HOLDINGS == left_backup
    assert RIGHT_HOLDINGS == right_backup


def test_comparison_does_not_modify_input_dataframes():
    """DataFrame을 넘겨받아도 원본 표가 바뀌지 않습니다."""
    left = pd.DataFrame(LEFT_HOLDINGS)
    right = pd.DataFrame(RIGHT_HOLDINGS)
    left_backup = left.copy(deep=True)
    right_backup = right.copy(deep=True)

    compare_institution_portfolios(left, right)

    pd.testing.assert_frame_equal(left, left_backup)
    pd.testing.assert_frame_equal(right, right_backup)


# ---------------------------------------------------------------------------
# 정렬 (compare_institution_portfolios)
# ---------------------------------------------------------------------------


def test_comparison_sorts_by_holding_type(comparison):
    """공통 보유 -> 기관 A 단독 -> 기관 B 단독 순서로 정렬합니다."""
    assert list(comparison[HOLDING_TYPE]) == [
        HOLDING_TYPE_COMMON,
        HOLDING_TYPE_COMMON,
        HOLDING_TYPE_LEFT_ONLY,
        HOLDING_TYPE_RIGHT_ONLY,
    ]


def test_comparison_sorts_by_larger_weight_within_type(comparison):
    """같은 유형 안에서는 두 기관 비중 중 큰 값이 큰 종목이 앞에 옵니다."""
    common = comparison[comparison[HOLDING_TYPE] == HOLDING_TYPE_COMMON]

    # APPLE은 기관 A 60%, COCA COLA는 기관 B 50%가 각각 최대값입니다.
    assert list(common["cusip"]) == [APPLE, COCA_COLA]


# ---------------------------------------------------------------------------
# 비교 요약 (summarize_institution_comparison)
# ---------------------------------------------------------------------------


def test_summary_counts_each_holding_type(comparison):
    """유형별 종목 수와 합집합 개수를 셉니다."""
    summary = summarize_institution_comparison(comparison)

    assert summary["common_count"] == 2
    assert summary["left_only_count"] == 1
    assert summary["right_only_count"] == 1
    assert summary["union_count"] == 4


def test_summary_calculates_security_overlap(comparison):
    """종목 중복률은 공통 종목 수 / 전체 고유 종목 수 * 100 입니다."""
    summary = summarize_institution_comparison(comparison)

    # 공통 2종목 / 전체 4종목 = 50%
    assert summary["security_overlap_pct"] == pytest.approx(50.0)


def test_summary_calculates_weighted_overlap(comparison):
    """비중 중복도는 공통 종목마다 작은 쪽 비중을 더한 값입니다."""
    summary = summarize_institution_comparison(comparison)

    # APPLE min(60, 25) = 25, COCA COLA min(30, 50) = 30 -> 55%
    assert summary["weighted_overlap_pct"] == pytest.approx(55.0)


def test_identical_portfolios_overlap_fully():
    """완전히 같은 두 포트폴리오는 종목 중복률과 비중 중복도가 모두 100%입니다."""
    comparison = compare_institution_portfolios(LEFT_HOLDINGS, LEFT_HOLDINGS)
    summary = summarize_institution_comparison(comparison)

    assert summary["security_overlap_pct"] == pytest.approx(100.0)
    assert summary["weighted_overlap_pct"] == pytest.approx(100.0)
    assert summary["left_only_count"] == 0
    assert summary["right_only_count"] == 0


def test_identical_weights_with_different_amounts_overlap_fully():
    """금액 규모가 달라도 비중 구성이 같으면 비중 중복도는 100%입니다."""
    left = [
        holding(APPLE, "APPLE INC", reported_value=600, shares=60),
        holding(COCA_COLA, "COCA COLA CO", reported_value=400, shares=40),
    ]
    # 금액은 100배지만 비중 구성(60% / 40%)은 같습니다.
    right = [
        holding(APPLE, "APPLE INC", reported_value=60000, shares=6000),
        holding(COCA_COLA, "COCA COLA CO", reported_value=40000, shares=4000),
    ]

    summary = summarize_institution_comparison(
        compare_institution_portfolios(left, right)
    )

    assert summary["weighted_overlap_pct"] == pytest.approx(100.0)


def test_completely_different_portfolios_do_not_overlap():
    """겹치는 종목이 하나도 없으면 두 중복도가 모두 0%입니다."""
    left = [holding(APPLE, "APPLE INC", reported_value=1000, shares=100)]
    right = [holding(MICROSOFT, "MICROSOFT CORP", reported_value=1000, shares=100)]

    summary = summarize_institution_comparison(
        compare_institution_portfolios(left, right)
    )

    assert summary["common_count"] == 0
    assert summary["union_count"] == 2
    assert summary["security_overlap_pct"] == pytest.approx(0.0)
    assert summary["weighted_overlap_pct"] == pytest.approx(0.0)


def test_summary_weighted_overlap_is_zero_when_total_value_is_zero():
    """전체 평가금액이 0이면 종목은 겹쳐도 비중 중복도는 0%입니다."""
    left = [holding(APPLE, "APPLE INC", reported_value=0, shares=100)]
    right = [holding(APPLE, "APPLE INC", reported_value=0, shares=50)]

    summary = summarize_institution_comparison(
        compare_institution_portfolios(left, right)
    )

    assert summary["security_overlap_pct"] == pytest.approx(100.0)
    assert summary["weighted_overlap_pct"] == pytest.approx(0.0)


def test_summary_handles_empty_comparison():
    """비교할 내용이 없으면 모든 개수와 비율이 0입니다."""
    summary = summarize_institution_comparison(compare_institution_portfolios([], []))

    assert summary == {
        "common_count": 0,
        "left_only_count": 0,
        "right_only_count": 0,
        "union_count": 0,
        "security_overlap_pct": 0.0,
        "weighted_overlap_pct": 0.0,
    }


def test_summary_handles_none():
    """표 대신 None을 넘겨도 오류 없이 0으로 채운 요약을 돌려줍니다."""
    summary = summarize_institution_comparison(None)

    assert summary["union_count"] == 0
    assert summary["security_overlap_pct"] == 0.0


def test_summary_percentages_stay_within_range(comparison):
    """중복률과 중복도는 0~100 범위를 벗어나지 않습니다."""
    summary = summarize_institution_comparison(comparison)

    assert 0.0 <= summary["security_overlap_pct"] <= 100.0
    assert 0.0 <= summary["weighted_overlap_pct"] <= 100.0
