"""services/llm_client.py 테스트.

실제 Gemini API에는 요청을 보내지 않습니다.
google.genai의 Client를 가짜(mock)로 바꿔치기하고, 예시 분석 결과로만 검증합니다.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.genai import errors as genai_errors

from services.llm_client import (
    TOP_CHANGE_COUNT,
    TOP_HOLDINGS_COUNT,
    EmptyResponseError,
    LlmApiError,
    build_briefing_prompt,
    generate_briefing,
)
from services.portfolio_analysis import compare_holdings, summarize_comparison

# 테스트용 가짜 API 키. 실제 키가 아니며, 오류 메시지에 새어 나가지 않는지 확인할 때 씁니다.
FAKE_API_KEY = "AIza-TEST-FAKE-KEY-1234567890"
FAKE_MODEL = "gemini-test-model"

# 예시 공시 정보.
CURRENT_FILING = {"filing_date": "2025-08-14", "report_date": "2025-06-30"}
PREVIOUS_FILING = {"filing_date": "2025-05-15", "report_date": "2025-03-31"}

# 예시 보유 종목.
#   APPLE       보유수량 100 -> 120 : 보유 확대
#   COCA COLA   보유수량  50 ->  40 : 보유 축소
#   BANK OF AM. 현재 분기에 없음    : 전량 매도
#   OCCIDENTAL  이전 분기에 없음    : 신규 편입
PREVIOUS_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 1000,
        "shares": 100,
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 500,
        "shares": 50,
    },
    {
        "issuer_name": "BANK OF AMERICA CORP",
        "class_title": "COM",
        "cusip": "060505104",
        "reported_value": 300,
        "shares": 30,
    },
]

CURRENT_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 1500,
        "shares": 120,
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 400,
        "shares": 40,
    },
    {
        "issuer_name": "OCCIDENTAL PETROLEUM CORP",
        "class_title": "COM",
        "cusip": "674599105",
        "reported_value": 700,
        "shares": 70,
    },
]


@pytest.fixture
def comparison() -> pd.DataFrame:
    """예시 두 분기 비교 결과."""
    return compare_holdings(PREVIOUS_HOLDINGS, CURRENT_HOLDINGS)


@pytest.fixture
def summary(comparison) -> dict:
    """예시 비교 결과의 요약값."""
    return summarize_comparison(comparison)


def make_response(text):
    """Gemini 응답을 흉내 낸 가짜 객체를 만듭니다."""
    response = MagicMock()
    response.text = text
    return response


def patch_client(response=None, error=None):
    """genai.Client를 가짜로 바꿔치기하는 patch 객체를 만듭니다.

    response를 주면 그 응답을 돌려주고, error를 주면 호출 시 그 오류를 냅니다.
    """
    client = MagicMock()
    if error is not None:
        client.models.generate_content.side_effect = error
    else:
        client.models.generate_content.return_value = response

    return patch("services.llm_client.genai.Client", return_value=client), client


# ---------------------------------------------------------------------------
# 프롬프트 만들기
# ---------------------------------------------------------------------------


def test_프롬프트에_공시_날짜가_들어간다(comparison, summary):
    prompt = build_briefing_prompt(
        comparison, summary, CURRENT_FILING, PREVIOUS_FILING, "Berkshire Hathaway"
    )

    assert "2025-06-30" in prompt  # 현재 분기 기준일
    assert "2025-08-14" in prompt  # 현재 분기 제출일
    assert "2025-03-31" in prompt  # 이전 분기 기준일
    assert "2025-05-15" in prompt  # 이전 분기 제출일
    assert "Berkshire Hathaway" in prompt


def test_프롬프트에_핵심_요약_수치가_들어간다(comparison, summary):
    prompt = build_briefing_prompt(
        comparison, summary, CURRENT_FILING, PREVIOUS_FILING
    )

    # 전체 평가금액: 이전 1,800 -> 현재 2,600 (증감 +800, 증감률 +44.44%)
    assert f"{summary['current_total_value']:,.0f}" in prompt
    assert f"{summary['previous_total_value']:,.0f}" in prompt
    assert f"{summary['total_value_change']:+,.0f}" in prompt
    assert f"{summary['total_value_change_pct']:+.2f}%" in prompt

    # 종목 수 변화가 요약값 그대로 들어가는지 확인합니다.
    assert "신규 편입: 1개" in prompt
    assert "전량 매도: 1개" in prompt
    assert "보유 확대: 1개" in prompt
    assert "보유 축소: 1개" in prompt


def test_프롬프트에_주요_변화_종목이_들어간다(comparison, summary):
    prompt = build_briefing_prompt(
        comparison, summary, CURRENT_FILING, PREVIOUS_FILING
    )

    assert "OCCIDENTAL PETROLEUM CORP" in prompt  # 신규 편입
    assert "BANK OF AMERICA CORP" in prompt  # 전량 매도
    assert "APPLE INC" in prompt  # 상위 보유 종목


def test_프롬프트에_생성_규칙이_들어간다(comparison, summary):
    prompt = build_briefing_prompt(
        comparison, summary, CURRENT_FILING, PREVIOUS_FILING
    )

    assert "핵심 요약" in prompt
    assert "해석 시 유의사항" in prompt
    assert "투자 추천" in prompt
    assert "주가 변화가 함께 반영" in prompt


def test_보유_종목이_많아도_프롬프트에_무제한으로_담기지_않는다():
    """종목이 60개여도 상위 목록 상한만큼만 프롬프트에 들어가야 합니다."""
    previous = [
        {
            "issuer_name": f"COMPANY {index:03d}",
            "class_title": "COM",
            "cusip": f"{index:09d}",
            "reported_value": 1000 - index,
            "shares": 100 - index,
        }
        for index in range(60)
    ]
    # 현재 분기에는 보유수량을 하나씩 늘려 모든 종목이 '보유 확대'가 되게 합니다.
    current = [
        {**row, "reported_value": row["reported_value"] + 10, "shares": row["shares"] + 1}
        for row in previous
    ]

    comparison = compare_holdings(previous, current)
    summary = summarize_comparison(comparison)
    prompt = build_briefing_prompt(comparison, summary, CURRENT_FILING, PREVIOUS_FILING)

    # 종목 한 줄에는 CUSIP이 한 번 나옵니다. 목록은 상위 보유 10개 + 변화 4종류 * 5개가 상한입니다.
    max_rows = TOP_HOLDINGS_COUNT + TOP_CHANGE_COUNT * 4
    assert prompt.count("(CUSIP ") <= max_rows

    # 중간 순위 종목은 어느 목록에도 들지 않으므로 프롬프트에 없어야 합니다.
    # (상위 목록에도, 비중 변화 상·하위 목록에도 들지 않는 위치입니다.)
    assert "COMPANY 030" not in prompt


def test_비교_결과가_비어도_프롬프트를_만들_수_있다():
    comparison = compare_holdings([], [])
    summary = summarize_comparison(comparison)

    prompt = build_briefing_prompt(comparison, summary, {}, {})

    assert "해당 종목 없음" in prompt
    assert "확인 불가" in prompt  # 날짜가 없을 때의 표시


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------


def test_정상_응답의_텍스트를_돌려준다():
    patcher, client = patch_client(response=make_response("## 1. 핵심 요약\n내용입니다."))

    with patcher:
        result = generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert result == "## 1. 핵심 요약\n내용입니다."

    # 모델명과 프롬프트만 전달하고, 생성 옵션은 넘기지 않습니다.
    client.models.generate_content.assert_called_once_with(
        model=FAKE_MODEL, contents="프롬프트"
    )


def test_빈_응답이면_오류를_낸다():
    patcher, _ = patch_client(response=make_response("   "))

    with patcher, pytest.raises(EmptyResponseError) as error:
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert "빈 응답" in str(error.value)


def test_응답_텍스트가_None이면_오류를_낸다():
    patcher, _ = patch_client(response=make_response(None))

    with patcher, pytest.raises(EmptyResponseError):
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)


def test_인증_오류를_안내_메시지로_바꾼다():
    api_error = genai_errors.ClientError(
        401,
        {"error": {"message": f"API key not valid: {FAKE_API_KEY}", "status": "UNAUTHENTICATED"}},
    )
    patcher, _ = patch_client(error=api_error)

    with patcher, pytest.raises(LlmApiError) as error:
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert "인증에 실패" in str(error.value)


def test_사용량_초과_오류를_안내_메시지로_바꾼다():
    api_error = genai_errors.ClientError(
        429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    patcher, _ = patch_client(error=api_error)

    with patcher, pytest.raises(LlmApiError) as error:
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert "사용량 한도" in str(error.value)


def test_서버_오류를_안내_메시지로_바꾼다():
    api_error = genai_errors.ServerError(
        503, {"error": {"message": "Service unavailable", "status": "UNAVAILABLE"}}
    )
    patcher, _ = patch_client(error=api_error)

    with patcher, pytest.raises(LlmApiError) as error:
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert "잠시 후 다시 시도" in str(error.value)


def test_네트워크_오류를_안내_메시지로_바꾼다():
    patcher, _ = patch_client(error=ConnectionError("connection refused"))

    with patcher, pytest.raises(LlmApiError) as error:
        generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert "연결하지 못했습니다" in str(error.value)


def test_오류_메시지에_API_키가_들어가지_않는다():
    """오류 메시지에 원본 문구가 섞여도 API 키가 새어 나가면 안 됩니다."""
    errors_to_check = [
        genai_errors.ClientError(401, {"error": {"message": f"key={FAKE_API_KEY}"}}),
        genai_errors.ClientError(429, {"error": {"message": f"key={FAKE_API_KEY}"}}),
        genai_errors.ServerError(500, {"error": {"message": f"key={FAKE_API_KEY}"}}),
        ConnectionError(f"failed to connect with key {FAKE_API_KEY}"),
    ]

    for api_error in errors_to_check:
        patcher, _ = patch_client(error=api_error)

        with patcher, pytest.raises(LlmApiError) as error:
            generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

        assert FAKE_API_KEY not in str(error.value)
        assert FAKE_API_KEY not in repr(error.value)


def test_반환값에_API_키가_들어가지_않는다():
    patcher, _ = patch_client(response=make_response("브리핑 본문입니다."))

    with patcher:
        result = generate_briefing("프롬프트", api_key=FAKE_API_KEY, model_name=FAKE_MODEL)

    assert FAKE_API_KEY not in result


def test_프롬프트에_API_키가_들어가지_않는다(comparison, summary):
    prompt = build_briefing_prompt(comparison, summary, CURRENT_FILING, PREVIOUS_FILING)

    assert FAKE_API_KEY not in prompt


@pytest.mark.parametrize(
    "prompt, api_key, model_name",
    [
        ("", FAKE_API_KEY, FAKE_MODEL),
        ("프롬프트", "", FAKE_MODEL),
        ("프롬프트", FAKE_API_KEY, ""),
    ],
)
def test_값이_비어_있으면_호출하지_않고_오류를_낸다(prompt, api_key, model_name):
    patcher, client = patch_client(response=make_response("응답"))

    with patcher, pytest.raises(ValueError):
        generate_briefing(prompt, api_key=api_key, model_name=model_name)

    client.models.generate_content.assert_not_called()
