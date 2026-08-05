"""기관 간 포트폴리오 비교 모듈.

역할:
    서로 다른 두 기관투자자가 '같은 분기(report_date)'에 제출한 13F 보유 종목을
    나란히 놓고 비교하는 계산을 맡습니다. 외부 API를 호출하지 않고, Streamlit에도
    의존하지 않는 순수 계산 로직만 담습니다.

현재 구현된 기능:
    - 공시 목록을 기준일(report_date)로 찾아 쓰기 좋게 정리 (index_filings_by_report_date)
    - 두 기관 모두 공시한 분기 찾기 (find_common_report_dates)
    - 한 기관의 보유 종목을 비중까지 계산해 정리 (prepare_institution_portfolio)
    - 두 기관 포트폴리오를 포지션 단위로 맞춰 비교 (compare_institution_portfolios)
    - 비교 결과 요약: 종목 중복률과 비중 중복도 (summarize_institution_comparison)

주의:
    - 포지션을 구분하는 기준은 portfolio_analysis 모듈의 '포지션 키'
      (CUSIP + put_call + share_type)를 그대로 씁니다. 키를 만드는 규칙과 중복 행을
      합산하는 규칙은 이 모듈에서 다시 구현하지 않고 aggregate_holdings를 호출합니다.
    - 공시 평가금액(reported_value)의 단위는 해석하거나 환산하지 않습니다.
      기관마다 금액 규모가 다르므로 비교는 '비중(%)'을 기준으로 합니다.
    - 이 모듈은 보유 사실만 계산합니다. 왜 샀는지·팔았는지 같은 투자 의도는
      공시 데이터로 알 수 없으므로 추정하지 않습니다.
"""

from __future__ import annotations

import pandas as pd

from services.portfolio_analysis import POSITION_KEY, aggregate_holdings

# 두 기관을 비교한 결과에서 각 줄이 어떤 경우인지 나타내는 이름.
# 화면에 그대로 표시할 수 있도록 한국어로 둡니다.
HOLDING_TYPE_COMMON = "공통 보유"
HOLDING_TYPE_LEFT_ONLY = "기관 A 단독"
HOLDING_TYPE_RIGHT_ONLY = "기관 B 단독"

# 결과 표를 정렬할 때 사용하는 유형 순서(공통 보유를 맨 위에 둡니다).
HOLDING_TYPE_ORDER = [
    HOLDING_TYPE_COMMON,
    HOLDING_TYPE_LEFT_ONLY,
    HOLDING_TYPE_RIGHT_ONLY,
]

# 각 줄이 어떤 유형인지 담는 열 이름.
HOLDING_TYPE = "holding_type"

# 한 기관의 포트폴리오를 정리한 표의 열 순서.
PORTFOLIO_COLUMNS = [
    POSITION_KEY,
    "issuer_name",
    "class_title",
    "cusip",
    "put_call",
    "share_type",
    "reported_value",
    "shares",
    "portfolio_weight_pct",
]

# 두 기관 비교 결과 표의 열 순서.
INSTITUTION_COMPARISON_COLUMNS = [
    POSITION_KEY,
    "issuer_name",
    "class_title",
    "cusip",
    "put_call",
    "share_type",
    "left_reported_value",
    "right_reported_value",
    "left_shares",
    "right_shares",
    "left_weight_pct",
    "right_weight_pct",
    "weight_gap_pct_point",
    HOLDING_TYPE,
]

# 각 열의 자료형. 데이터가 비어 있을 때도 표의 형태를 일정하게 유지합니다.
_PORTFOLIO_DTYPES = {
    POSITION_KEY: "object",
    "issuer_name": "object",
    "class_title": "object",
    "cusip": "object",
    "put_call": "object",
    "share_type": "object",
    "reported_value": "float64",
    "shares": "float64",
    "portfolio_weight_pct": "float64",
}

_COMPARISON_DTYPES = {
    POSITION_KEY: "object",
    "issuer_name": "object",
    "class_title": "object",
    "cusip": "object",
    "put_call": "object",
    "share_type": "object",
    "left_reported_value": "float64",
    "right_reported_value": "float64",
    "left_shares": "float64",
    "right_shares": "float64",
    "left_weight_pct": "float64",
    "right_weight_pct": "float64",
    "weight_gap_pct_point": "float64",
    HOLDING_TYPE: "object",
}

# 비교 결과에서 두 기관의 종목 정보를 합칠 때 대상이 되는 글자 열.
_TEXT_COLUMNS = ["issuer_name", "class_title", "cusip", "put_call", "share_type"]

# 비율(%)이 가질 수 있는 범위.
_MIN_PERCENT = 0.0
_MAX_PERCENT = 100.0


def index_filings_by_report_date(filings) -> dict[str, dict]:
    """공시 목록을 기준일(report_date)로 찾아볼 수 있는 딕셔너리로 바꿉니다.

    두 기관을 비교하려면 '같은 분기'의 공시를 짝지어야 합니다. 그런데 공시 목록은
    제출일(filing_date) 순서로 내려오므로, 분기 말일인 기준일(report_date)을
    열쇠로 삼아 바로 찾을 수 있게 정리해 둡니다.

    Args:
        filings: sec_client.get_recent_13f_filings가 돌려준 공시 목록.
            각 항목은 report_date, filing_date 등을 담은 딕셔너리입니다.

    Returns:
        {기준일 문자열: 공시 딕셔너리} 형태의 딕셔너리.
        원본을 나중에 바꿔도 영향이 없도록 각 공시는 복사해서 담습니다.

    Note:
        - 기준일이 비어 있는 공시는 어느 분기 것인지 알 수 없으므로 제외합니다.
        - 같은 기준일로 여러 건이 있으면(정정 제출 등) 제출일이 가장 늦은 것을
          고릅니다. 제출일까지 같으면 목록에서 나중에 나온 것을 씁니다.
    """
    indexed: dict[str, dict] = {}

    for filing in filings or []:
        # 딕셔너리가 아닌 값이 섞여 있어도 전체가 멈추지 않도록 건너뜁니다.
        if not isinstance(filing, dict):
            continue

        report_date = _clean_text(filing.get("report_date"))
        if not report_date:
            continue

        filing_date = _clean_text(filing.get("filing_date"))
        chosen = indexed.get(report_date)

        # ">=" 이므로 제출일이 같으면 나중에 나온 공시가 앞의 것을 덮어씁니다.
        if chosen is None or filing_date >= _clean_text(chosen.get("filing_date")):
            # 호출한 쪽의 딕셔너리가 결과를 통해 바뀌지 않도록 복사해 둡니다.
            indexed[report_date] = dict(filing)

    return indexed


def find_common_report_dates(left_filings, right_filings) -> list[str]:
    """두 기관이 모두 공시한 분기(기준일)를 최신순으로 찾습니다.

    비교의 기준은 제출일이 아니라 기준일입니다. 제출일은 기관마다 며칠씩 다르지만,
    기준일(분기 말일)이 같아야 같은 시점의 포트폴리오를 비교하는 것이 됩니다.

    Args:
        left_filings: 기관 A의 공시 목록.
        right_filings: 기관 B의 공시 목록.

    Returns:
        두 기관 모두 공시가 있는 기준일 문자열 목록(최신순).
        겹치는 분기가 없으면 빈 목록을 돌려줍니다.
    """
    left_index = index_filings_by_report_date(left_filings)
    right_index = index_filings_by_report_date(right_filings)

    common = set(left_index) & set(right_index)

    # 기준일은 "2025-06-30"처럼 자리수가 고정된 형식이라 글자 순 정렬이 곧 날짜 순입니다.
    return sorted(common, reverse=True)


def prepare_institution_portfolio(holdings) -> pd.DataFrame:
    """한 기관의 보유 종목을 포지션별 한 줄로 정리하고 비중을 계산합니다.

    중복 행 합산과 포지션 키 생성은 portfolio_analysis.aggregate_holdings에
    맡깁니다. 이 함수는 거기에 '전체 대비 비중(%)'만 더합니다.

    Args:
        holdings: sec_client.get_13f_holdings가 돌려준 보유 종목 목록.
            pandas DataFrame을 넘겨도 되며, 어느 경우든 원본은 바뀌지 않습니다.

    Returns:
        PORTFOLIO_COLUMNS 순서를 가진 표. 보유 종목이 없으면 같은 열을 가진
        빈 표를 돌려줍니다.

    Note:
        비중은 reported_value / 전체 reported_value * 100 입니다.
        전체 금액이 0이거나 계산할 수 없으면 모든 비중을 0으로 둡니다.
        (금액 단위는 기관마다 같은 공시 규칙을 따르므로, 비중은 단위와 무관하게
        서로 비교할 수 있습니다.)
    """
    aggregated = aggregate_holdings(holdings)

    if aggregated.empty:
        return _empty_portfolio()

    # aggregate_holdings가 이미 숫자로 바꾸고 결측치를 0으로 채워 두지만,
    # 이 함수만 따로 보아도 규칙이 분명하도록 한 번 더 명시합니다.
    reported_value = _numeric_column(aggregated, "reported_value")
    shares = _numeric_column(aggregated, "shares")

    portfolio = pd.DataFrame(
        {
            POSITION_KEY: aggregated[POSITION_KEY].astype("object"),
            "issuer_name": aggregated["issuer_name"].astype("object"),
            "class_title": aggregated["class_title"].astype("object"),
            "cusip": aggregated["cusip"].astype("object"),
            "put_call": aggregated["put_call"].astype("object"),
            "share_type": aggregated["share_type"].astype("object"),
            "reported_value": reported_value,
            "shares": shares,
            "portfolio_weight_pct": _weight_percent(reported_value),
        }
    )

    return portfolio[PORTFOLIO_COLUMNS].reset_index(drop=True)


def compare_institution_portfolios(left_holdings, right_holdings) -> pd.DataFrame:
    """두 기관의 같은 분기 포트폴리오를 포지션 단위로 나란히 놓고 비교합니다.

    한쪽만 보유한 종목도 빠지지 않도록 outer join(양쪽을 모두 살리는 합치기)을
    사용합니다. 짝을 맞추는 기준은 CUSIP 하나가 아니라 포지션 키
    (CUSIP + put_call + share_type)이므로, 같은 회사라도 보통주 보유와
    Put/Call 옵션 보유는 서로 다른 줄로 남습니다.

    Args:
        left_holdings: 기관 A의 보유 종목 목록(또는 DataFrame).
        right_holdings: 기관 B의 보유 종목 목록(또는 DataFrame).

    Returns:
        INSTITUTION_COMPARISON_COLUMNS 순서를 가진 표.
        정렬은 공통 보유 -> 기관 A 단독 -> 기관 B 단독 순이고,
        각 유형 안에서는 두 기관 비중 중 큰 값이 큰 종목부터 놓입니다.
        두 입력이 모두 비어 있으면 같은 열을 가진 빈 표를 돌려줍니다.

    Note:
        weight_gap_pct_point는 left_weight_pct - right_weight_pct(퍼센트포인트)입니다.
        양수면 기관 A가 더 많이 담은 종목, 음수면 기관 B가 더 많이 담은 종목입니다.
        한쪽만 보유한 종목의 반대쪽 금액·수량·비중은 0으로 채웁니다.
    """
    left = prepare_institution_portfolio(left_holdings)
    right = prepare_institution_portfolio(right_holdings)

    if left.empty and right.empty:
        return _empty_comparison()

    merged = left.merge(
        right,
        on=POSITION_KEY,
        how="outer",
        suffixes=("_left", "_right"),
        # 어느 쪽에 있던 줄인지 pandas가 알려주는 표시(both/left_only/right_only).
        indicator="_side",
    )

    result = pd.DataFrame(
        {
            POSITION_KEY: _text_column(merged, POSITION_KEY),
            "left_reported_value": _numeric_column(merged, "reported_value_left"),
            "right_reported_value": _numeric_column(merged, "reported_value_right"),
            "left_shares": _numeric_column(merged, "shares_left"),
            "right_shares": _numeric_column(merged, "shares_right"),
            "left_weight_pct": _numeric_column(merged, "portfolio_weight_pct_left"),
            "right_weight_pct": _numeric_column(merged, "portfolio_weight_pct_right"),
        }
    )

    # 종목 정보는 한쪽에만 적혀 있을 수 있으므로 비어 있지 않은 값을 씁니다.
    for column in _TEXT_COLUMNS:
        result[column] = _prefer_non_empty_text(merged, column)

    result["weight_gap_pct_point"] = (
        result["left_weight_pct"] - result["right_weight_pct"]
    )
    result[HOLDING_TYPE] = _classify_sides(merged["_side"])

    return _sort_comparison(result)


def summarize_institution_comparison(comparison: pd.DataFrame) -> dict:
    """두 기관 비교 결과를 한눈에 보는 요약값으로 정리합니다.

    Args:
        comparison: compare_institution_portfolios가 돌려준 표.

    Returns:
        다음 키를 가진 딕셔너리.
            common_count: 두 기관이 함께 보유한 포지션 수
            left_only_count: 기관 A만 보유한 포지션 수
            right_only_count: 기관 B만 보유한 포지션 수
            union_count: 두 기관 포지션을 합친 고유 개수
            security_overlap_pct: 종목 중복률(%).
                공통 포지션 수 / 전체 고유 포지션 수 * 100
            weighted_overlap_pct: 비중 중복도(%).
                공통 포지션마다 두 기관 비중 중 작은 값을 더한 값

    Note:
        종목 중복률은 '몇 종목이나 같이 들고 있나'를, 비중 중복도는 '포트폴리오의
        몇 %가 겹치나'를 나타냅니다. 소수 종목에 크게 투자한 기관과 여러 종목에
        조금씩 투자한 기관은 두 값이 크게 다를 수 있어 함께 봅니다.
        비교할 내용이 없으면 모든 값이 0입니다.
    """
    empty_summary = {
        "common_count": 0,
        "left_only_count": 0,
        "right_only_count": 0,
        "union_count": 0,
        "security_overlap_pct": 0.0,
        "weighted_overlap_pct": 0.0,
    }

    if comparison is None or len(comparison) == 0:
        return empty_summary

    if HOLDING_TYPE not in comparison.columns:
        return empty_summary

    counts = comparison[HOLDING_TYPE].value_counts()
    common_count = int(counts.get(HOLDING_TYPE_COMMON, 0))
    left_only_count = int(counts.get(HOLDING_TYPE_LEFT_ONLY, 0))
    right_only_count = int(counts.get(HOLDING_TYPE_RIGHT_ONLY, 0))
    union_count = common_count + left_only_count + right_only_count

    if union_count > 0:
        security_overlap_pct = _clamp_percent(common_count / union_count * 100)
    else:
        security_overlap_pct = 0.0

    common_rows = comparison[comparison[HOLDING_TYPE] == HOLDING_TYPE_COMMON]
    if common_rows.empty:
        weighted_overlap_pct = 0.0
    else:
        # 겹치는 만큼만 세려면 두 비중 중 작은 값을 더합니다.
        # (예: A가 10%, B가 4%면 겹치는 부분은 4%입니다.)
        overlap = pd.concat(
            [
                _numeric_column(common_rows, "left_weight_pct"),
                _numeric_column(common_rows, "right_weight_pct"),
            ],
            axis=1,
        ).min(axis=1)
        # 음수 비중은 정상 데이터에서 나오지 않지만, 섞여 들어와도 합계를 깎지
        # 않도록 0 미만은 0으로 봅니다.
        weighted_overlap_pct = _clamp_percent(float(overlap.clip(lower=0.0).sum()))

    return {
        "common_count": common_count,
        "left_only_count": left_only_count,
        "right_only_count": right_only_count,
        "union_count": union_count,
        "security_overlap_pct": security_overlap_pct,
        "weighted_overlap_pct": weighted_overlap_pct,
    }


# ---------------------------------------------------------------------------
# 내부 보조 함수 (모듈 밖에서 직접 쓰지 않습니다)
# ---------------------------------------------------------------------------


def _clean_text(value) -> str:
    """값을 앞뒤 공백이 없는 문자열로 바꿉니다. 빈 값은 빈 문자열이 됩니다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_text(values: pd.Series) -> pd.Series:
    """열의 값을 앞뒤 공백이 없는 문자열로 바꿉니다. 빈 값은 빈 문자열입니다."""
    text = values.astype("object").where(values.notna(), "")
    return text.astype(str).str.strip()


def _text_column(table: pd.DataFrame, name: str) -> pd.Series:
    """표에서 글자 열을 꺼냅니다. 열이 없으면 빈 문자열로 채웁니다."""
    if name not in table.columns:
        return pd.Series([""] * len(table), index=table.index, dtype="object")
    return _as_text(table[name])


def _numeric_column(table: pd.DataFrame, name: str) -> pd.Series:
    """표에서 숫자 열을 꺼냅니다.

    한쪽 기관에만 있는 종목은 반대쪽 값이 비어 있으므로 0으로 채웁니다.
    (그 기관은 해당 종목을 보유하지 않았다는 뜻입니다.)
    """
    if name not in table.columns:
        return pd.Series(0.0, index=table.index, dtype="float64")
    return pd.to_numeric(table[name], errors="coerce").fillna(0.0).astype("float64")


def _prefer_non_empty_text(merged: pd.DataFrame, name: str) -> pd.Series:
    """기관 A의 표기를 우선하고, 비어 있으면 기관 B의 표기를 씁니다."""
    left = _text_column(merged, f"{name}_left")
    right = _text_column(merged, f"{name}_right")
    return left.where(left != "", right)


def _weight_percent(values: pd.Series) -> pd.Series:
    """전체 합계 대비 비중(%)을 계산합니다.

    합계가 0이거나 숫자로 다룰 수 없으면 모든 비중을 0으로 둡니다.
    (0으로 나누어 무한대나 NaN이 생기지 않게 하기 위한 처리입니다.)
    """
    total = float(values.sum())

    if not pd.notna(total) or total <= 0:
        return pd.Series(0.0, index=values.index, dtype="float64")

    return (values / total * 100).astype("float64")


def _classify_sides(sides: pd.Series) -> pd.Series:
    """merge 표시(both/left_only/right_only)를 한국어 유형 이름으로 바꿉니다."""
    mapping = {
        "both": HOLDING_TYPE_COMMON,
        "left_only": HOLDING_TYPE_LEFT_ONLY,
        "right_only": HOLDING_TYPE_RIGHT_ONLY,
    }
    return sides.astype(str).map(mapping).astype("object")


def _sort_comparison(result: pd.DataFrame) -> pd.DataFrame:
    """유형 순서와 비중 크기에 따라 결과를 정렬하고 열 순서를 맞춥니다."""
    ordered = result.copy()

    # 정렬에만 쓰는 임시 열입니다. 결과 표에는 남기지 않습니다.
    ordered["_type_order"] = pd.Categorical(
        ordered[HOLDING_TYPE], categories=HOLDING_TYPE_ORDER, ordered=True
    )
    ordered["_max_weight"] = ordered[["left_weight_pct", "right_weight_pct"]].max(axis=1)

    # 비중이 같은 줄의 순서까지 항상 같게 하려고 마지막 기준으로 포지션 키를 둡니다.
    ordered = ordered.sort_values(
        ["_type_order", "_max_weight", POSITION_KEY],
        ascending=[True, False, True],
    )

    return ordered[INSTITUTION_COMPARISON_COLUMNS].reset_index(drop=True)


def _clamp_percent(value: float) -> float:
    """비율이 0~100 범위를 벗어나지 않게 잘라 냅니다."""
    number = float(value)

    if not pd.notna(number):
        return 0.0

    return min(max(number, _MIN_PERCENT), _MAX_PERCENT)


def _empty_portfolio() -> pd.DataFrame:
    """보유 종목이 없을 때 쓰는, 열만 있는 빈 표."""
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in _PORTFOLIO_DTYPES.items()}
    )[PORTFOLIO_COLUMNS]


def _empty_comparison() -> pd.DataFrame:
    """비교할 데이터가 없을 때 쓰는, 열만 있는 빈 표."""
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in _COMPARISON_DTYPES.items()}
    )[INSTITUTION_COMPARISON_COLUMNS]
