"""포트폴리오 분석 모듈.

역할:
    sec_client가 가져온 13F 보유 종목 데이터를 받아 숫자를 계산하고
    사람이 이해하기 쉬운 표 형태로 정리하는 책임을 맡습니다.
    외부 API를 호출하지 않는 순수 계산 로직만 담습니다.

현재 구현된 기능:
    - 포지션 식별용 키 생성 (build_position_key)
    - 한 분기의 보유 종목을 포지션 키 기준으로 합산 (aggregate_holdings)
    - 두 분기(이전/현재)를 비교해 포지션별 변화 계산 (compare_holdings)
    - 비교 결과를 한눈에 보는 요약값 계산 (summarize_comparison)

포함 예정 기능:
    - 포트폴리오 집중도(상위 N개 종목 비중 합계) 계산

주의:
    - 포지션을 구분하는 기준은 CUSIP(증권 고유 번호) + put_call + share_type을
      묶은 '포지션 키'입니다. CUSIP만으로는 같은 종목의 보통주 보유와 Put/Call
      옵션 보유가 한 줄로 합쳐지고, 주식 수(SH)와 원금액(PRN)처럼 단위가 다른
      수량까지 더해져 버립니다. 회사명은 공시마다 표기가 조금씩 달라질 수 있어
      CUSIP이 있는 경우에는 식별 기준으로 쓰지 않습니다.
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
TEXT_INPUT_COLUMNS = ["cusip", "issuer_name", "class_title", "put_call", "share_type"]
NUMERIC_INPUT_COLUMNS = ["reported_value", "shares"]
INPUT_COLUMNS = TEXT_INPUT_COLUMNS + NUMERIC_INPUT_COLUMNS

# 포지션을 구분하는 키가 담기는 열 이름. 결과 표에도 남겨 검증에 쓸 수 있게 합니다.
POSITION_KEY = "position_key"

# 포지션 키 구성 요소를 정규화할 때 쓰는 기본값.
# 일반 주식(옵션이 아닌 보유)은 put_call 칸이 비어 있으므로 EQUITY로 통일합니다.
PUT_CALL_EQUITY = "EQUITY"
# 수량 단위(SH=주식 수, PRN=원금액)가 비어 있으면 UNKNOWN으로 통일합니다.
SHARE_TYPE_UNKNOWN = "UNKNOWN"

# 포지션 키 구성 요소를 잇는 구분자.
POSITION_KEY_SEPARATOR = "|"

# CUSIP이 비어 있어 발행사명으로 대체 식별한 포지션 키에 붙는 접두어.
# 이 접두어로 '정상 CUSIP 기준 키'와 '대체(fallback) 키'를 구분할 수 있습니다.
FALLBACK_KEY_PREFIX = "NOCUSIP:"

# 포지션 키 기준으로 합산한 표의 열.
AGGREGATED_COLUMNS = [POSITION_KEY] + INPUT_COLUMNS

# compare_holdings가 돌려주는 표의 열 순서.
COMPARISON_COLUMNS = [
    POSITION_KEY,
    "cusip",
    "issuer_name",
    "class_title",
    "put_call",
    "share_type",
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
    POSITION_KEY: "object",
    "cusip": "object",
    "issuer_name": "object",
    "class_title": "object",
    "put_call": "object",
    "share_type": "object",
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


def build_position_key(
    cusip="",
    put_call="",
    share_type="",
    issuer_name="",
    class_title="",
) -> str:
    """한 보유 행(포지션)을 식별하는 키를 만듭니다.

    같은 회사(같은 CUSIP)라도 보통주 보유와 Call 옵션, Put 옵션은 성격이
    전혀 다른 포지션입니다. 수량 단위(SH=주식 수, PRN=원금액)도 다르면
    더할 수 없습니다. 그래서 이 세 가지를 함께 묶어 하나의 키로 만듭니다.

    기본 형태 (CUSIP이 있을 때):
        "037833100|EQUITY|SH"   보통주 보유
        "037833100|CALL|SH"     Call 옵션 보유
        "037833100|PUT|SH"      Put 옵션 보유

    대체 형태 (CUSIP이 비어 있을 때):
        "NOCUSIP:APPLE INC|COM|EQUITY|SH"

    Args:
        cusip: CUSIP(증권 고유 번호).
        put_call: Put 또는 Call 표기. 비어 있으면 EQUITY로 봅니다.
        share_type: 수량 단위(SH/PRN). 비어 있으면 UNKNOWN으로 봅니다.
        issuer_name: 발행사명. CUSIP이 비어 있을 때만 키에 사용합니다.
        class_title: 증권 종류. CUSIP이 비어 있을 때만 키에 사용합니다.

    Returns:
        사람이 읽을 수 있는 포지션 키 문자열. 앞뒤 공백을 없애고 대문자로
        통일한 값들을 이어 붙입니다.

    Note:
        CUSIP이 없는 행끼리 하나로 합쳐지면 서로 다른 종목이 섞이므로,
        그 경우에는 발행사명·증권 종류까지 키에 넣고 FALLBACK_KEY_PREFIX를
        붙여 대체 키임을 표시합니다(is_fallback_position_key로 확인).
    """
    normalized_cusip = _normalize_key_text(cusip)
    normalized_put_call = _normalize_key_text(put_call) or PUT_CALL_EQUITY
    normalized_share_type = _normalize_key_text(share_type) or SHARE_TYPE_UNKNOWN

    if normalized_cusip:
        parts = [normalized_cusip, normalized_put_call, normalized_share_type]
    else:
        # CUSIP이 없으면 발행사명·증권 종류로 대체 식별합니다.
        parts = [
            FALLBACK_KEY_PREFIX + _normalize_key_text(issuer_name),
            _normalize_key_text(class_title),
            normalized_put_call,
            normalized_share_type,
        ]

    return POSITION_KEY_SEPARATOR.join(parts)


def is_fallback_position_key(position_key) -> bool:
    """CUSIP 없이 발행사명으로 만든 대체 키인지 알려줍니다.

    데이터 품질을 점검할 때(CUSIP이 빠진 공시 행이 얼마나 되는지) 사용합니다.
    """
    if position_key is None or (
        isinstance(position_key, float) and pd.isna(position_key)
    ):
        return False
    return str(position_key).startswith(FALLBACK_KEY_PREFIX)


def aggregate_holdings(holdings) -> pd.DataFrame:
    """한 분기의 보유 종목을 포지션 키 기준으로 하나의 줄로 합칩니다.

    같은 포지션이 여러 줄로 나뉘어 공시되는 경우가 있습니다(운용 주체나 의결권
    구분이 다를 때). 비중을 계산하려면 포지션별로 한 줄이어야 하므로,
    평가금액과 보유수량을 더해 하나로 만듭니다.

    합치는 기준은 CUSIP 하나가 아니라 build_position_key가 만든 키
    (CUSIP + put_call + share_type)입니다. 따라서 같은 회사의 보통주 보유와
    Put/Call 옵션 보유는 서로 다른 줄로 남습니다.

    Args:
        holdings: sec_client.get_13f_holdings가 돌려준 딕셔너리 목록.
            pandas DataFrame을 넘겨도 됩니다. 어느 경우든 원본은 바뀌지 않습니다.

    Returns:
        AGGREGATED_COLUMNS 순서(position_key, cusip, issuer_name, class_title,
        put_call, share_type, reported_value, shares)를 가진 표.
        입력이 비어 있으면 같은 열을 가진 빈 표를 돌려줍니다.

    Note:
        - 숫자로 바꿀 수 없는 평가금액·보유수량은 0으로 처리합니다.
        - 글자 항목(발행사명, 증권 종류 등)은 비어 있지 않은 첫 번째 값을 대표값으로
          씁니다. 같은 포지션의 어떤 줄에만 값이 적혀 있어도 빈칸이 되지 않습니다.
        - CUSIP이 비어 있는 비정상 데이터는 서로 다른 종목이 한 줄로 합쳐지지
          않도록 발행사명·증권 종류를 대체 식별자로 사용합니다.
    """
    table = _to_input_frame(holdings)

    if table.empty:
        return _empty_aggregated()

    grouped = table.groupby(POSITION_KEY, as_index=False, sort=False).agg(
        cusip=("cusip", _first_valid_text),
        issuer_name=("issuer_name", _first_valid_text),
        class_title=("class_title", _first_valid_text),
        put_call=("put_call", _first_valid_text),
        share_type=("share_type", _first_valid_text),
        reported_value=("reported_value", "sum"),
        shares=("shares", "sum"),
    )

    return grouped[AGGREGATED_COLUMNS]


def compare_holdings(previous_holdings, current_holdings) -> pd.DataFrame:
    """이전 분기와 현재 분기의 보유 종목을 비교합니다.

    두 분기 중 한쪽에만 있는 포지션(신규 편입 / 전량 매도)도 빠지지 않도록
    outer join(양쪽 모두 살리는 합치기)을 사용합니다.

    짝을 맞추는 기준은 CUSIP 하나가 아니라 포지션 키
    (CUSIP + put_call + share_type)입니다. 그래서 이전 분기의 Call 옵션과
    현재 분기의 보통주 보유가 같은 줄로 잘못 묶이지 않습니다.

    Args:
        previous_holdings: 이전 분기(과거) 공시의 보유 종목 목록.
        current_holdings: 현재 분기(최근) 공시의 보유 종목 목록.

    Returns:
        포지션별 한 줄씩 담긴 표. 열은 COMPARISON_COLUMNS 순서와 같습니다.
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

    # 두 분기를 포지션 키 기준으로 맞춥니다. 키는 aggregate_holdings가 이미
    # 같은 규칙으로 만들어 두었으므로 그대로 씁니다.
    merged = previous_table.merge(
        current_table,
        on=POSITION_KEY,
        how="outer",
        suffixes=("_previous", "_current"),
    )

    result = pd.DataFrame(
        {
            POSITION_KEY: _text_column(merged, POSITION_KEY),
            # 종목 정보는 최근 공시 표기를 우선하고, 없으면 이전 공시 표기를 씁니다.
            "cusip": _prefer_current_text(merged, "cusip"),
            "issuer_name": _prefer_current_text(merged, "issuer_name"),
            "class_title": _prefer_current_text(merged, "class_title"),
            "put_call": _prefer_current_text(merged, "put_call"),
            "share_type": _prefer_current_text(merged, "share_type"),
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

    # 금액이 같은 줄의 순서까지 항상 같게 하려고, 마지막 기준으로 포지션 키를 둡니다.
    sorted_result = result.sort_values(
        ["current_reported_value", "previous_reported_value", "cusip", POSITION_KEY],
        ascending=[False, False, True, True],
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
        return pd.DataFrame(columns=[POSITION_KEY] + INPUT_COLUMNS)

    # 없는 열이 있어도 계산이 멈추지 않도록 빈 값으로 채워 둡니다.
    for column in INPUT_COLUMNS:
        if column not in table.columns:
            table[column] = "" if column in TEXT_INPUT_COLUMNS else 0

    for column in TEXT_INPUT_COLUMNS:
        table[column] = _as_text(table[column])

    for column in NUMERIC_INPUT_COLUMNS:
        # 숫자로 바꿀 수 없는 값(빈칸, 글자 등)은 0으로 처리합니다.
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    table[POSITION_KEY] = _position_key_column(table)

    return table[[POSITION_KEY] + INPUT_COLUMNS]


def _position_key_column(table: pd.DataFrame) -> pd.Series:
    """표의 각 줄에 대해 포지션 키를 만들어 하나의 열로 돌려줍니다.

    키를 만드는 규칙은 build_position_key 한 곳에만 두어, 한 줄씩 계산할 때와
    표 단위로 계산할 때의 결과가 어긋나지 않게 합니다.
    """
    if table.empty:
        return pd.Series(dtype="object")

    return table.apply(
        lambda row: build_position_key(
            cusip=row["cusip"],
            put_call=row["put_call"],
            share_type=row["share_type"],
            issuer_name=row["issuer_name"],
            class_title=row["class_title"],
        ),
        axis=1,
    ).astype("object")


def _normalize_key_text(value) -> str:
    """포지션 키에 넣을 값을 정규화합니다.

    앞뒤 공백과 중간의 겹친 공백을 정리하고 대문자로 통일합니다.
    빈 값(None, NaN)은 빈 문자열이 됩니다. 이렇게 해야 " sh"와 "SH"처럼
    표기만 다른 값이 서로 다른 포지션으로 갈라지지 않습니다.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split()).upper()


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
            POSITION_KEY: pd.Series(dtype="object"),
            "cusip": pd.Series(dtype="object"),
            "issuer_name": pd.Series(dtype="object"),
            "class_title": pd.Series(dtype="object"),
            "put_call": pd.Series(dtype="object"),
            "share_type": pd.Series(dtype="object"),
            "reported_value": pd.Series(dtype="float64"),
            "shares": pd.Series(dtype="float64"),
        }
    )[AGGREGATED_COLUMNS]


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
