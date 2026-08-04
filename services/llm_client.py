"""Gemini API 연동 모듈.

역할:
    portfolio_analysis가 만들어 낸 분석 결과(숫자와 표)를 Gemini API에 전달해
    비전문가도 이해할 수 있는 자연어 브리핑으로 바꾸는 책임을 맡습니다.

현재 구현된 기능:
    - 분석 결과를 프롬프트 문장으로 변환 (build_briefing_prompt)
    - Gemini API 호출과 응답 텍스트 반환 (generate_briefing)

중요한 원칙:
    - 숫자 계산과 종목 분류는 모두 portfolio_analysis가 이미 끝낸 결과만 씁니다.
      이 모듈은 그 결과를 문장으로 옮겨 적을 뿐이고, Gemini에게도 새로운 계산이나
      추측을 시키지 않습니다.
    - 토큰(요청 크기)을 아끼기 위해 전체 보유 종목 표는 보내지 않고,
      상위 종목만 골라서 보냅니다.

주의:
    - API 키와 모델명은 이 모듈이 직접 읽지 않고, 호출하는 쪽(화면)이 st.secrets에서
      읽어 함수 인자로 넘겨 줍니다. 이 모듈은 st.secrets를 참조하지 않습니다.
    - 오류 메시지에 API 키가 섞여 나가지 않도록, 원본 오류 문구는 그대로 쓰지 않고
      정해진 한국어 안내 문구로만 바꿉니다.
    - LLM이 만든 설명에는 사실과 다른 내용이 섞일 수 있으므로 화면에 주의 문구를
      함께 표시합니다.
"""

from __future__ import annotations

import pandas as pd
from google import genai
from google.genai import errors as genai_errors

from services.portfolio_analysis import STATUS_EXITED, STATUS_NEW

# 프롬프트에 넣을 현재 분기 상위 종목 수. 전체 표를 보내지 않기 위한 상한입니다.
TOP_HOLDINGS_COUNT = 10

# 비중 증가/감소, 신규 편입, 전량 매도 목록에 넣을 종목 수 상한.
TOP_CHANGE_COUNT = 5

# 프롬프트에서 값이 없을 때 쓰는 표시.
UNKNOWN_TEXT = "확인 불가"

# HTTP 상태 코드. 어떤 오류인지 구분해 안내 문구를 고르는 데 씁니다.
_UNAUTHORIZED_CODES = (401, 403)
_QUOTA_CODE = 429


class LlmApiError(RuntimeError):
    """Gemini API 호출이 실패했을 때 쓰는 오류.

    화면에서 그대로 보여 줄 수 있도록 한국어 안내 문구만 담습니다.
    원본 오류 메시지(주소나 인증 정보가 섞일 수 있음)는 담지 않습니다.
    """


class EmptyResponseError(LlmApiError):
    """API 호출은 성공했지만 돌려받은 글이 비어 있을 때 쓰는 오류.

    LlmApiError를 상속하므로, LLM 관련 오류를 한꺼번에 처리하는 코드에서도 잡힙니다.
    """


# ---------------------------------------------------------------------------
# 프롬프트 만들기
# ---------------------------------------------------------------------------

# Gemini에게 주는 역할과 규칙. 데이터와 분리해 두어 읽기 쉽게 했습니다.
_INSTRUCTIONS = """당신은 미국 SEC 13F 공시 데이터를 일반 독자에게 설명하는 리서치 어시스턴트입니다.

아래 [분석 데이터]는 Python 프로그램이 13F 공시 원문에서 이미 계산해 둔 결과입니다.
당신의 역할은 이 결과를 한국어로 설명하고 요약하는 것뿐입니다.

반드시 지킬 규칙:
- [분석 데이터]에 없는 사실이나 숫자를 만들어내지 마세요. 새로 계산하지도 마세요.
- 데이터에 없는 종목명, 산업 분류, 주가, 시장 전망을 덧붙이지 마세요.
- 평가금액 변화를 실제 매수·매도 금액이라고 단정하지 마세요.
- 평가금액의 단위를 '천 달러'나 '달러'라고 단정하거나 추정하지 마세요. 금액은 SEC
  Information Table의 reported value 값이므로, 단위를 붙이지 말고 숫자를 그대로 쓰거나
  '공시 평가금액'이라고만 표현하세요. 임의로 단위를 곱하거나 나누어 환산하지 마세요.
- 평가금액 변화에는 보유수량 변화와 주가 변화가 함께 반영된다는 점을 본문에 밝혀 주세요.
- 투자 추천, 목표주가, 매수·매도 의견을 쓰지 마세요.
- 이 글이 교육용 분석이라는 점을 밝혀 주세요.
- 전문 용어는 짧게 풀어서 설명해 주세요.
- 모든 문장을 한국어로 쓰세요.

다음 5개 항목을 이 순서와 제목 그대로, Markdown 소제목(##)으로 작성하세요.
## 1. 핵심 요약
## 2. 포트폴리오 규모와 집중도 변화
## 3. 주요 신규 편입 및 전량 매도
## 4. 비중 확대 및 축소 종목
## 5. 해석 시 유의사항
"""


def build_briefing_prompt(
    comparison,
    summary,
    current_filing=None,
    previous_filing=None,
    manager_name="분석 대상 기관투자자",
) -> str:
    """두 분기 분석 결과를 Gemini에 보낼 프롬프트 문자열로 바꿉니다.

    Args:
        comparison: portfolio_analysis.compare_holdings가 돌려준 표.
        summary: portfolio_analysis.summarize_comparison이 돌려준 요약 딕셔너리.
        current_filing: 현재 분기 공시 정보(filing_date, report_date를 씁니다).
        previous_filing: 이전 분기 공시 정보.
        manager_name: 분석 대상 운용사 이름.

    Returns:
        Gemini에 그대로 넘길 수 있는 프롬프트 문자열.

    Note:
        전체 보유 종목 표는 넣지 않습니다. 현재 분기 상위 TOP_HOLDINGS_COUNT개와
        변화 상위 TOP_CHANGE_COUNT개 목록만 넣어 요청 크기를 제한합니다.
    """
    table = _to_comparison_frame(comparison)
    summary = summary or {}
    current_filing = current_filing or {}
    previous_filing = previous_filing or {}

    sections = [
        _INSTRUCTIONS,
        "[분석 데이터]",
        f"분석 대상: {manager_name}",
        "",
        "## 비교 기준",
        f"- 현재 분기 기준일(report_date): {_text(current_filing.get('report_date'))}",
        f"- 현재 분기 제출일(filing_date): {_text(current_filing.get('filing_date'))}",
        f"- 이전 분기 기준일(report_date): {_text(previous_filing.get('report_date'))}",
        f"- 이전 분기 제출일(filing_date): {_text(previous_filing.get('filing_date'))}",
        "",
        "## 전체 규모",
        f"- 현재 분기 전체 평가금액(reported value): {_number(summary.get('current_total_value'))}",
        f"- 이전 분기 전체 평가금액(reported value): {_number(summary.get('previous_total_value'))}",
        f"- 증감액: {_number(summary.get('total_value_change'), signed=True)}",
        f"- 증감률: {_percent(summary.get('total_value_change_pct'))}",
        "",
        "## 종목 수 변화",
        f"- 신규 편입: {_count(summary.get('new_position_count'))}개",
        f"- 전량 매도: {_count(summary.get('exited_position_count'))}개",
        f"- 보유 확대: {_count(summary.get('increased_position_count'))}개",
        f"- 보유 축소: {_count(summary.get('decreased_position_count'))}개",
        f"- 유지: {_count(summary.get('unchanged_position_count'))}개",
        "",
        f"## 현재 분기 평가금액 상위 {TOP_HOLDINGS_COUNT}개 종목",
        _format_rows(_top_holdings(table)),
        "",
        f"## 비중 증가 상위 {TOP_CHANGE_COUNT}개 종목",
        _format_rows(_top_weight_increases(table)),
        "",
        f"## 비중 감소 상위 {TOP_CHANGE_COUNT}개 종목",
        _format_rows(_top_weight_decreases(table)),
        "",
        f"## 신규 편입 상위 {TOP_CHANGE_COUNT}개 종목",
        _format_rows(_top_new_positions(table)),
        "",
        f"## 전량 매도 상위 {TOP_CHANGE_COUNT}개 종목",
        _format_rows(_top_exited_positions(table)),
        "",
        "값 출처 안내: 평가금액은 SEC Information Table의 reported value 필드 값을 "
        "환산 없이 그대로 사용한 것입니다. 이 데이터에는 금액 단위가 명시되어 있지 "
        "않으므로, 단위를 추정하지 말고 숫자만 그대로 인용하세요. "
        "비중은 %, 비중 변화는 %포인트(%p)입니다.",
        "",
        "위 [분석 데이터]만 사용해 앞에서 지시한 5개 항목의 브리핑을 작성하세요.",
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------


def generate_briefing(prompt: str, api_key: str, model_name: str) -> str:
    """Gemini API에 프롬프트를 보내고 생성된 글만 돌려줍니다.

    Args:
        prompt: build_briefing_prompt가 만든 프롬프트 문자열.
        api_key: Gemini API 키. 호출하는 쪽이 st.secrets에서 읽어 넘겨 줍니다.
        model_name: 사용할 모델 이름.

    Returns:
        모델이 생성한 글(문자열).

    Raises:
        ValueError: 프롬프트, API 키, 모델명이 비어 있는 경우.
        EmptyResponseError: 호출은 됐지만 돌려받은 글이 비어 있는 경우.
        LlmApiError: 인증 실패, 사용량 초과, 서버 오류, 통신 실패 등.

    Note:
        temperature, top_p, top_k 같은 생성 옵션은 설정하지 않고 모델 기본값을 씁니다.
        오류 메시지에는 API 키가 절대 포함되지 않습니다.
    """
    if not prompt or not str(prompt).strip():
        raise ValueError("Gemini에 보낼 프롬프트가 비어 있습니다.")
    if not api_key or not str(api_key).strip():
        raise ValueError("Gemini API 키가 비어 있습니다.")
    if not model_name or not str(model_name).strip():
        raise ValueError("Gemini 모델 이름이 비어 있습니다.")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except genai_errors.APIError as error:
        # API가 상태 코드를 돌려준 경우. 원본 문구 대신 정해진 안내로 바꿉니다.
        raise LlmApiError(_api_error_message(error)) from None
    except Exception as error:
        # 통신 실패 등 그 밖의 오류. 오류 종류 이름만 남기고 상세 내용은 버립니다.
        raise LlmApiError(
            "Gemini API에 연결하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요. "
            f"(오류 종류: {type(error).__name__})"
        ) from None

    text = getattr(response, "text", None)
    if text is None or not str(text).strip():
        raise EmptyResponseError(
            "Gemini가 빈 응답을 보냈습니다. 잠시 후 다시 시도하거나, "
            "모델 설정을 확인해 주세요."
        )

    return str(text).strip()


def _api_error_message(error) -> str:
    """Gemini API 오류를 사용자에게 보여 줄 한국어 안내로 바꿉니다.

    원본 오류 메시지에는 요청 주소나 인증 정보가 섞여 있을 수 있어 쓰지 않고,
    상태 코드만 보고 미리 준비한 문구를 고릅니다.
    """
    code = getattr(error, "code", None)

    if code in _UNAUTHORIZED_CODES:
        return (
            "Gemini API 인증에 실패했습니다. "
            "`.streamlit/secrets.toml`의 GEMINI_API_KEY가 올바른지 확인해 주세요."
        )
    if code == _QUOTA_CODE:
        return (
            "Gemini API 사용량 한도를 넘었습니다. "
            "잠시 기다렸다가 다시 시도해 주세요."
        )
    if isinstance(code, int) and 500 <= code < 600:
        return (
            "Gemini 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요. "
            f"(응답 코드: {code})"
        )
    if code == 404:
        return (
            "요청한 Gemini 모델을 찾을 수 없습니다. "
            "`.streamlit/secrets.toml`의 GEMINI_MODEL 값을 확인해 주세요."
        )

    code_text = code if code is not None else "확인 불가"
    return (
        "Gemini API 호출이 실패했습니다. 잠시 후 다시 시도해 주세요. "
        f"(응답 코드: {code_text})"
    )


# ---------------------------------------------------------------------------
# 내부 보조 함수 (모듈 밖에서 직접 쓰지 않습니다)
# ---------------------------------------------------------------------------


def _to_comparison_frame(comparison) -> pd.DataFrame:
    """비교 결과를 표 형태로 맞춥니다. 원본은 바꾸지 않습니다."""
    if isinstance(comparison, pd.DataFrame):
        return comparison.copy()
    return pd.DataFrame(list(comparison or []))


def _top_holdings(table: pd.DataFrame) -> pd.DataFrame:
    """현재 분기 평가금액이 큰 상위 종목을 고릅니다."""
    if table.empty or "current_reported_value" not in table.columns:
        return table.head(0)

    held = table[table["current_reported_value"] > 0]
    return held.sort_values("current_reported_value", ascending=False).head(
        TOP_HOLDINGS_COUNT
    )


def _top_weight_increases(table: pd.DataFrame) -> pd.DataFrame:
    """비중이 늘어난 상위 종목을 고릅니다."""
    if table.empty or "weight_change_pct_point" not in table.columns:
        return table.head(0)

    increased = table[table["weight_change_pct_point"] > 0]
    return increased.sort_values("weight_change_pct_point", ascending=False).head(
        TOP_CHANGE_COUNT
    )


def _top_weight_decreases(table: pd.DataFrame) -> pd.DataFrame:
    """비중이 줄어든 상위 종목을 고릅니다(가장 많이 줄어든 순서)."""
    if table.empty or "weight_change_pct_point" not in table.columns:
        return table.head(0)

    decreased = table[table["weight_change_pct_point"] < 0]
    return decreased.sort_values("weight_change_pct_point", ascending=True).head(
        TOP_CHANGE_COUNT
    )


def _top_new_positions(table: pd.DataFrame) -> pd.DataFrame:
    """신규 편입 종목 중 현재 평가금액이 큰 순서로 고릅니다."""
    rows = _rows_with_status(table, STATUS_NEW)
    if rows.empty:
        return rows
    return rows.sort_values("current_reported_value", ascending=False).head(
        TOP_CHANGE_COUNT
    )


def _top_exited_positions(table: pd.DataFrame) -> pd.DataFrame:
    """전량 매도 종목 중 이전 평가금액이 컸던 순서로 고릅니다."""
    rows = _rows_with_status(table, STATUS_EXITED)
    if rows.empty:
        return rows
    return rows.sort_values("previous_reported_value", ascending=False).head(
        TOP_CHANGE_COUNT
    )


def _rows_with_status(table: pd.DataFrame, status: str) -> pd.DataFrame:
    """변화 구분이 같은 줄만 골라 냅니다."""
    if table.empty or "change_status" not in table.columns:
        return table.head(0)
    return table[table["change_status"] == status]


def _format_rows(rows: pd.DataFrame) -> str:
    """고른 종목들을 프롬프트에 넣을 목록 문장으로 바꿉니다."""
    if rows is None or rows.empty:
        return "- 해당 종목 없음"

    return "\n".join(_format_row(row) for _, row in rows.iterrows())


def _format_row(row) -> str:
    """종목 한 줄을 사람이 읽는 문장으로 바꿉니다. 값은 계산 결과 그대로 씁니다."""
    name = _text(row.get("issuer_name")) or _text(row.get("cusip"))
    return (
        f"- {name} (CUSIP {_text(row.get('cusip'))}): "
        f"현재 평가금액 {_number(row.get('current_reported_value'))}, "
        f"이전 평가금액 {_number(row.get('previous_reported_value'))}, "
        f"현재 비중 {_percent(row.get('current_weight'), signed=False)}, "
        f"이전 비중 {_percent(row.get('previous_weight'), signed=False)}, "
        f"비중 변화 {_percent(row.get('weight_change_pct_point'))}p, "
        f"보유수량 변화 {_number(row.get('shares_change'), signed=True)}주, "
        f"구분 {_text(row.get('change_status'))}"
    )


def _text(value) -> str:
    """값을 글자로 바꿉니다. 비어 있으면 '확인 불가'로 표시합니다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_TEXT
    text = str(value).strip()
    return text if text else UNKNOWN_TEXT


def _number(value, signed: bool = False) -> str:
    """숫자를 천 단위 쉼표가 있는 글자로 바꿉니다. 계산할 수 없으면 안내 문구."""
    number = pd.to_numeric(value, errors="coerce")
    if number is None or pd.isna(number):
        return UNKNOWN_TEXT
    return f"{float(number):+,.0f}" if signed else f"{float(number):,.0f}"


def _percent(value, signed: bool = True) -> str:
    """비율을 소수점 두 자리 글자로 바꿉니다. 계산할 수 없으면 안내 문구."""
    number = pd.to_numeric(value, errors="coerce")
    if number is None or pd.isna(number):
        return UNKNOWN_TEXT
    return f"{float(number):+.2f}%" if signed else f"{float(number):.2f}%"


def _count(value) -> str:
    """종목 수를 정수 글자로 바꿉니다. 값이 없으면 0으로 둡니다."""
    number = pd.to_numeric(value, errors="coerce")
    if number is None or pd.isna(number):
        return "0"
    return f"{int(number)}"
