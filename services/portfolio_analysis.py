"""포트폴리오 분석 모듈.

역할:
    sec_client가 가져온 13F 보유 종목 데이터를 받아 숫자를 계산하고
    사람이 이해하기 쉬운 표 형태로 정리하는 책임을 맡습니다.
    외부 API를 호출하지 않는 순수 계산 로직만 담습니다.

현재 구현된 기능:
    - 한 분기의 보유 종목을 CUSIP 기준으로 합산 (aggregate_holdings)
    - 두 분기(이전/현재)를 비교해 종목별 변화 계산 (compare_holdings)
    - 비교 결과를 한눈에 보는 요약값 계산 (summarize_comparison)

포함 예정 기능:
    - 포트폴리오 집중도(상위 N개 종목 비중 합계) 계산

주의:
    - 종목을 구분하는 기준은 CUSIP(증권 고유 번호)입니다. 회사명은 공시마다
      표기가 조금씩 달라질 수 있어 식별 기준으로 쓰지 않습니다.
    - 공시 평가금액(reported_value)은 SEC Information Table의 reported value 필드
      값을 그대로 사용합니다. 이 모듈은 단위를 해석하거나 환산하지 않습니다.
      비중(%)은 같은 분기 안에서의 비율이므로 단위와 무관하게 유효합니다.
"""

from __future__ import annotations

import pandas as pd

# 변화 구분에 사용하는 이름. 화면에 그대로 표시할 수 있도록 한국어로 둡니다.
STATUS_NEW = "신규 편입"
STATUS_EXITED = "전량 매도"
STATUS_INCREASED = "보유 확대"
STATUS_DECREASED = "보유 축소"
STATUS_UNCHANGED = "유지"

# sec_client.get_13f_holdings가 돌려주는 키 중 이 모듈이 사용하는 것들.
TEXT_INPUT_COLUMNS = ["cusip", "issuer_name", "class_title"]
NUMERIC_INPUT_COLUMNS = ["reported_value", "shares"]
INPUT_COLUMNS = TEXT_INPUT_COLUMNS + NUMERIC_INPUT_COLUMNS

# CUSIP 기준으로 합산한 표의 열.
AGGREGATED_COLUMNS = INPUT_COLUMNS

# compare_holdings가 돌려주는 표의 열 순서.
COMPARISON_COLUMNS = [
    "cusip",
    "issuer_name",
    "class_title",
    "previous_reported_value",
    "current_reported_value",
    "reported_value_change",
    "value_change_pct",
    "previous_shares",
    "current_shares",
    "shares_change",
    "previous_weight",
    "current_weight",
    "weight_change_pct_point",
    "change_status",
]

# 각 열을 어떤 자료형으로 둘지. 데이터가 비어 있을 때도 형태를 일정하게 유지합니다.
_COMPARISON_DTYPES = {
    "cusip": "object",
    "issuer_name": "object",
    "class_title": "object",
    "previous_reported_value": "float64",
    "current_reported_value": "float64",
    "reported_value_change": "float64",
    "value_change_pct": "float64",
    "previous_shares": "float64",
    "current_shares": "float64",
    "shares_change": "float64",
    "previous_weight": "float64",
    "current_weight": "float64",
    "weight_change_pct_point": "float64",
    "change_status": "object",
}

# 종목을 묶는 임시 열 이름. 결과 표에는 남기지 않습니다.
_GROUP_KEY = "_group_key"


def aggregate_holdings(holdings) -> pd.DataFrame:
    """한 분기의 보유 종목을 CUSIP 기준으로 하나의 줄로 합칩니다.

    같은 종목이 여러 줄로 나뉘어 공시되는 경우가 있습니다(운용 주체나 의결권
    구분이 다를 때). 비중을 계산하려면 종목별로 한 줄이어야 하므로,
    평가금액과 보유수량을 더해 하나로 만듭니다.

    Args:
        holdings: sec_client.get_13f_holdings가 돌려준 딕셔너리 목록.
            pandas DataFrame을 넘겨도 됩니다. 어느 경우든 원본은 바뀌지 않습니다.

    Returns:
        cusip, issuer_name, class_title, reported_value, shares 열을 가진 표.
        입력이 비어 있으면 같은 열을 가진 빈 표를 돌려줍니다.

    Note:
        - 숫자로 바꿀 수 없는 평가금액·보유수량은 0으로 처리합니다.
        - issuer_name과 class_title은 비어 있지 않은 첫 번째 값을 씁니다.
        - CUSIP이 비어 있는 비정상 데이터는 서로 다른 종목이 한 줄로 합쳐지지
          않도록 발행사명을 임시 식별자로 사용합니다.
    """
    table = _to_input_frame(holdings)

    if table.empty:
        return _empty_aggregated()

    grouped = table.groupby(_GROUP_KEY, as_index=False, sort=False).agg(
        cusip=("cusip", _first_valid_text),
        issuer_name=("issuer_name", _first_valid_text),
        class_title=("class_title", _first_valid_text),
        reported_value=("reported_value", "sum"),
        shares=("shares", "sum"),
    )

    return grouped


def compare_holdings(previous_holdings, current_holdings) -> pd.DataFrame:
    """이전 분기와 현재 분기의 보유 종목을 비교합니다.

    두 분기 중 한쪽에만 있는 종목(신규 편입 / 전량 매도)도 빠지지 않도록
    outer join(양쪽 모두 살리는 합치기)을 사용합니다.

    Args:
        previous_holdings: 이전 분기(과거) 공시의 보유 종목 목록.
        current_holdings: 현재 분기(최근) 공시의 보유 종목 목록.

    Returns:
        종목별 한 줄씩 담긴 표. 열은 COMPARISON_COLUMNS 순서와 같습니다.
        현재 분기 평가금액이 큰 종목부터 정렬되며, 전량 매도 종목은
        현재 금액이 0이므로 뒤쪽에 놓입니다.
        입력이 모두 비어 있으면 같은 열을 가진 빈 표를 돌려줍니다.

    Note:
        비중(previous_weight, current_weight)은 각 분기의 전체 평가금액 대비
        비율(%)입니다. weight_change_pct_point는 두 비중의 차이(퍼센트포인트)입니다.
        이전 평가금액이 0이면 변화율(value_change_pct)은 계산할 수 없어 NaN입니다.
    """
    previous_table = aggregate_holdings(previous_holdings)
    current_table = aggregate_holdings(current_holdings)

    if previous_table.empty and current_table.empty:
        return _empty_comparison()

    # 두 분기를 CUSIP 기준으로 맞춥니다. CUSIP이 비어 있는 비정상 데이터는
    # aggregate_holdings와 같은 규칙(발행사명 기준)으로 짝을 찾습니다.
    merged = previous_table.assign(**{_GROUP_KEY: _build_group_key(previous_table)}).merge(
        current_table.assign(**{_GROUP_KEY: _build_group_key(current_table)}),
        on=_GROUP_KEY,
        how="outer",
        suffixes=("_previous", "_current"),
    )

    result = pd.DataFrame(
        {
            # 종목 정보는 최근 공시 표기를 우선하고, 없으면 이전 공시 표기를 씁니다.
            "cusip": _prefer_current_text(merged, "cusip"),
            "issuer_name": _prefer_current_text(merged, "issuer_name"),
            "class_title": _prefer_current_text(merged, "class_title"),
            "previous_reported_value": _numeric_column(
                merged, "reported_value_previous"
            ),
            "current_reported_value": _numeric_column(
                merged, "reported_value_current"
            ),
            "previous_shares": _numeric_column(merged, "shares_previous"),
            "current_shares": _numeric_column(merged, "shares_current"),
        }
    )

    result["reported_value_change"] = (
        result["current_reported_value"] - result["previous_reported_value"]
    )
    result["shares_change"] = result["current_shares"] - result["previous_shares"]

    # 0으로 나누지 않도록, 이전 평가금액이 0이면 NaN으로 바꿔 계산합니다.
    # NaN으로 나눈 결과도 NaN이 되므로 무한대(inf)가 생기지 않습니다.
    previous_value = result["previous_reported_value"]
    divisor = previous_value.where(previous_value > 0)
    result["value_change_pct"] = result["reported_value_change"] / divisor * 100

    result["previous_weight"] = _weight(result["previous_reported_value"])
    result["current_weight"] = _weight(result["current_reported_value"])
    result["weight_change_pct_point"] = (
        result["current_weight"] - result["previous_weight"]
    )

    result["change_status"] = _classify_changes(result)

    sorted_result = result.sort_values(
        ["current_reported_value", "previous_reported_value", "cusip"],
        ascending=[False, False, True],
    )

    return sorted_result[COMPARISON_COLUMNS].reset_index(drop=True)


def summarize_comparison(comparison: pd.DataFrame) -> dict:
    """비교 결과 표를 한눈에 보는 요약값으로 정리합니다.

    Args:
        comparison: compare_holdings가 돌려준 표.

    Returns:
        다음 키를 가진 딕셔너리.
            current_total_value: 현재 분기 전체 공시 평가금액
            previous_total_value: 이전 분기 전체 공시 평가금액
            total_value_change: 평가금액 증감
            total_value_change_pct: 평가금액 증감률(%).
                이전 합계가 0이면 계산할 수 없어 None입니다.
            new_position_count: 신규 편입 종목 수
            exited_position_count: 전량 매도 종목 수
            increased_position_count: 보유 확대 종목 수
            decreased_position_count: 보유 축소 종목 수
            unchanged_position_count: 유지 종목 수
    """
    if comparison is None or len(comparison) == 0:
        return {
            "current_total_value": 0.0,
            "previous_total_value": 0.0,
            "total_value_change": 0.0,
            "total_value_change_pct": None,
            "new_position_count": 0,
            "exited_position_count": 0,
            "increased_position_count": 0,
            "decreased_position_count": 0,
            "unchanged_position_count": 0,
        }

    current_total = _safe_sum(comparison, "current_reported_value")
    previous_total = _safe_sum(comparison, "previous_reported_value")
    total_change = current_total - previous_total

    if previous_total > 0:
        total_change_pct = total_change / previous_total * 100
    else:
        # 이전 분기 금액이 0이면 증감률을 계산할 수 없습니다.
        total_change_pct = None

    counts = comparison["change_status"].value_counts()

    return {
        "current_total_value": current_total,
        "previous_total_value": previous_total,
        "total_value_change": total_change,
        "total_value_change_pct": total_change_pct,
        "new_position_count": int(counts.get(STATUS_NEW, 0)),
        "exited_position_count": int(counts.get(STATUS_EXITED, 0)),
        "increased_position_count": int(counts.get(STATUS_INCREASED, 0)),
        "decreased_position_count": int(counts.get(STATUS_DECREASED, 0)),
        "unchanged_position_count": int(counts.get(STATUS_UNCHANGED, 0)),
    }


# ---------------------------------------------------------------------------
# 내부 보조 함수 (모듈 밖에서 직접 쓰지 않습니다)
# ---------------------------------------------------------------------------


def _to_input_frame(holdings) -> pd.DataFrame:
    """입력 데이터를 계산하기 좋은 표로 바꿉니다. 원본은 바꾸지 않습니다."""
    if isinstance(holdings, pd.DataFrame):
        # 넘겨받은 표를 그대로 고치면 호출한 쪽의 데이터가 바뀌므로 복사합니다.
        table = holdings.copy()
    else:
        # 딕셔너리 목록으로 새 표를 만듭니다. 원본 딕셔너리는 건드리지 않습니다.
        table = pd.DataFrame(list(holdings or []))

    if table.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS + [_GROUP_KEY])

    # 없는 열이 있어도 계산이 멈추지 않도록 빈 값으로 채워 둡니다.
    for column in INPUT_COLUMNS:
        if column not in table.columns:
            table[column] = "" if column in TEXT_INPUT_COLUMNS else 0

    for column in TEXT_INPUT_COLUMNS:
        table[column] = _as_text(table[column])

    for column in NUMERIC_INPUT_COLUMNS:
        # 숫자로 바꿀 수 없는 값(빈칸, 글자 등)은 0으로 처리합니다.
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    table[_GROUP_KEY] = _build_group_key(table)

    return table[INPUT_COLUMNS + [_GROUP_KEY]]


def _build_group_key(table: pd.DataFrame) -> pd.Series:
    """종목을 묶는 기준 값을 만듭니다. 기본은 CUSIP입니다.

    CUSIP이 비어 있는 비정상 데이터까지 한 줄로 합쳐 버리면 서로 다른 종목이
    섞이므로, 그 경우에만 발행사명과 증권 종류를 임시 식별자로 사용합니다.
    """
    cusip = table["cusip"]
    fallback = "이름기준:" + table["issuer_name"] + "|" + table["class_title"]
    return cusip.where(cusip != "", fallback)


def _as_text(values: pd.Series) -> pd.Series:
    """값을 앞뒤 공백이 없는 문자열로 바꿉니다. 빈 값은 빈 문자열이 됩니다."""
    text = values.astype("object").where(values.notna(), "")
    return text.astype(str).str.strip()


def _first_valid_text(values) -> str:
    """여러 값 중 비어 있지 않은 첫 번째 값을 돌려줍니다. 없으면 빈 문자열."""
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _empty_aggregated() -> pd.DataFrame:
    """보유 종목이 없을 때 쓰는, 열만 있는 빈 표."""
    return pd.DataFrame(
        {
            "cusip": pd.Series(dtype="object"),
            "issuer_name": pd.Series(dtype="object"),
            "class_title": pd.Series(dtype="object"),
            "reported_value": pd.Series(dtype="float64"),
            "shares": pd.Series(dtype="float64"),
        }
    )


def _empty_comparison() -> pd.DataFrame:
    """비교할 데이터가 없을 때 쓰는, 열만 있는 빈 표."""
    return pd.DataFrame(
        {column: pd.Series(dtype=dtype) for column, dtype in _COMPARISON_DTYPES.items()}
    )[COMPARISON_COLUMNS]


def _text_column(merged: pd.DataFrame, name: str) -> pd.Series:
    """합친 표에서 글자 열을 꺼냅니다. 열이 없으면 빈 문자열로 채웁니다."""
    if name not in merged.columns:
        return pd.Series([""] * len(merged), index=merged.index, dtype="object")
    return _as_text(merged[name])


def _prefer_current_text(merged: pd.DataFrame, name: str) -> pd.Series:
    """현재 분기 표기를 우선하고, 비어 있으면 이전 분기 표기를 씁니다."""
    current = _text_column(merged, f"{name}_current")
    previous = _text_column(merged, f"{name}_previous")
    return current.where(current != "", previous)


def _numeric_column(merged: pd.DataFrame, name: str) -> pd.Series:
    """합친 표에서 숫자 열을 꺼냅니다.

    한쪽 분기에만 있는 종목은 값이 비어 있으므로 0으로 채웁니다.
    (그 분기에는 보유하지 않았다는 뜻입니다.)
    """
    if name not in merged.columns:
        return pd.Series(0.0, index=merged.index, dtype="float64")
    return pd.to_numeric(merged[name], errors="coerce").fillna(0.0).astype("float64")


def _weight(values: pd.Series) -> pd.Series:
    """전체 합계 대비 비중(%)을 계산합니다. 합계가 0이면 모두 0으로 둡니다."""
    total = float(values.sum())
    if total > 0:
        return values / total * 100
    return pd.Series(0.0, index=values.index, dtype="float64")


def _classify_changes(result: pd.DataFrame) -> pd.Series:
    """종목별 변화 구분(신규 편입 / 전량 매도 / 확대 / 축소 / 유지)을 정합니다.

    '그 분기에 보유했는가'는 평가금액이나 보유수량 중 하나라도 0보다 크면
    보유한 것으로 봅니다. 확대·축소는 보유수량 변화로 판단합니다.

    신규 편입과 전량 매도를 나중에 덮어써서, 확대·축소보다 우선하게 합니다.
    """
    previously_held = (result["previous_reported_value"] > 0) | (
        result["previous_shares"] > 0
    )
    currently_held = (result["current_reported_value"] > 0) | (
        result["current_shares"] > 0
    )

    status = pd.Series(STATUS_UNCHANGED, index=result.index, dtype="object")
    status[result["shares_change"] > 0] = STATUS_INCREASED
    status[result["shares_change"] < 0] = STATUS_DECREASED
    status[previously_held & ~currently_held] = STATUS_EXITED
    status[~previously_held & currently_held] = STATUS_NEW

    return status


def _safe_sum(table: pd.DataFrame, column: str) -> float:
    """열의 합계를 실수로 돌려줍니다. 열이 없거나 값이 없으면 0입니다."""
    if column not in table.columns:
        return 0.0

    total = pd.to_numeric(table[column], errors="coerce").sum(skipna=True)
    return float(total) if pd.notna(total) else 0.0
