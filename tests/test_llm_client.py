"""services/llm_client.py 테스트.

실제 Gemini API에는 요청을 보내지 않습니다.
google.genai의 Client를 가짜(mock)로 바꿔치기하고, 예시 분석 결과로만 검증합니다.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.genai import errors as genai_errors

from services.institution_comparison import (
    build_institution_comparison_briefing_payload,
    compare_institution_portfolios,
    summarize_institution_comparison,
)
from services.llm_client import (
    TOP_CHANGE_COUNT,
    TOP_HOLDINGS_COUNT,
    EmptyResponseError,
    LlmApiError,
    build_briefing_prompt,
    build_institution_comparison_briefing_prompt,
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
# 기관 간 비교 프롬프트 만들기
#
# 프롬프트에는 institution_comparison이 계산해 둔 값만 들어가야 하고,
# 원본 XML이나 전체 비교 표는 들어가면 안 됩니다.
# ---------------------------------------------------------------------------

# 예시 두 기관의 보유 종목.
#   기관 A 합계 10000: APPLE 60% / COCA COLA 30% / AMEX 10%
#   기관 B 합계  4000: APPLE 25% / COCA COLA 50% / MICROSOFT 25%
LEFT_INSTITUTION_HOLDINGS = [
    {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "reported_value": 6000,
        "shares": 600,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "COCA COLA CO",
        "class_title": "COM",
        "cusip": "191216100",
        "reported_value": 3000,
        "shares": 300,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "AMERICAN EXPRESS CO",
        "class_title": "COM",
        "cusip": "025816109",
        "reported_value": 1000,
        "shares": 100,
        "share_type": "SH",
        "put_call": "",
    },
]

RIGHT_INSTITUTION_HOLDINGS = [
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
        "reported_value": 2000,
        "shares": 200,
        "share_type": "SH",
        "put_call": "",
    },
    {
        "issuer_name": "MICROSOFT CORP",
        "class_title": "COM",
        "cusip": "594918104",
        "reported_value": 1000,
        "shares": 50,
        "share_type": "SH",
        "put_call": "",
    },
]

LEFT_MANAGER_NAME = "Berkshire Hathaway"
RIGHT_MANAGER_NAME = "Pershing Square Capital Management"
INSTITUTION_REPORT_DATE = "2025-06-30"


@pytest.fixture
def institution_comparison() -> pd.DataFrame:
    """예시 두 기관의 같은 분기 비교 결과."""
    return compare_institution_portfolios(
        LEFT_INSTITUTION_HOLDINGS, RIGHT_INSTITUTION_HOLDINGS
    )


@pytest.fixture
def institution_payload(institution_comparison) -> dict:
    """예시 비교 결과로 만든 AI 브리핑 입력 데이터."""
    return build_institution_comparison_briefing_payload(
        institution_comparison,
        summarize_institution_comparison(institution_comparison),
        report_date=INSTITUTION_REPORT_DATE,
        left_manager_name=LEFT_MANAGER_NAME,
        right_manager_name=RIGHT_MANAGER_NAME,
    )


def test_기관_비교_프롬프트에_기관명과_기준일이_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert INSTITUTION_REPORT_DATE in prompt
    assert LEFT_MANAGER_NAME in prompt
    assert RIGHT_MANAGER_NAME in prompt


def test_기관_비교_프롬프트에_중복률과_중복도가_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)
    summary = institution_payload["summary"]

    assert "종목 중복률" in prompt
    assert "비중 기준 중복도" in prompt
    # 공통 2종목 / 전체 4종목 = 50.00%, min(60,25) + min(30,50) = 55.00%
    assert f"{summary['security_overlap_pct']:.2f}%" in prompt
    assert f"{summary['weighted_overlap_pct']:.2f}%" in prompt


def test_기관_비교_프롬프트에_종목_수_요약이_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert "공통 보유 종목 수: 2개" in prompt
    assert "기관 A 단독 보유 종목 수: 1개" in prompt
    assert "기관 B 단독 보유 종목 수: 1개" in prompt
    assert "두 기관 보유를 합친 고유 종목 수: 4개" in prompt


def test_기관_비교_프롬프트에_주요_종목과_비중이_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert "APPLE INC" in prompt  # 공통 보유
    assert "AMERICAN EXPRESS CO" in prompt  # 기관 A 단독
    assert "MICROSOFT CORP" in prompt  # 기관 B 단독
    # 비중은 계산된 값 그대로 들어가고, 비중 차이는 %포인트로 표시합니다.
    assert "기관 A 비중 60.00%" in prompt
    assert "기관 B 비중 25.00%" in prompt
    assert "비중 차이 +35.00%p" in prompt


def test_기관_비교_프롬프트에_투자_의도_추정_금지가_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    # 공통 보유를 같은 전략으로 단정하지 말라는 지시.
    assert "동일한 투자 전략" in prompt
    # 비중 차이를 매매나 선호의 증거로 단정하지 말라는 지시.
    assert "비중 차이를 매수·매도의 증거" in prompt
    # 투자 이유·향후 매매·주가 전망을 추정하지 말라는 지시.
    assert "투자 이유" in prompt
    assert "주가 전망을 추정하지 마세요" in prompt


def test_기관_비교_프롬프트에_투자_추천_금지가_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert "투자 추천이나 매수·매도 의견을 제공하지 마세요" in prompt


def test_기관_비교_프롬프트에_공시_시차와_범위_제한이_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    # 13F는 분기 말 기준이며 실시간 포트폴리오가 아니라는 점.
    assert "실시간 포트폴리오가 아니라는 점" in prompt
    # 13F 대상 증권 범위에 한정된다는 점.
    assert "13F 공시 대상 증권 범위에 한정" in prompt
    # 공시 이후 포트폴리오가 달라졌을 수 있다는 점.
    assert "공시 이후 실제 포트폴리오는 달라졌을 수 있다" in prompt
    # 수치가 없으면 추정하지 말라는 점.
    assert "추정하지 말고" in prompt


def test_기관_비교_프롬프트에_계산_금지와_출력_구조가_들어간다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert "직접 계산하지 마세요" in prompt
    assert "## 1. 한눈에 보는 비교" in prompt
    assert "## 2. 공통 보유 특징" in prompt
    assert "## 3. 기관별 차이" in prompt
    assert "## 4. 해석 시 주의사항" in prompt
    assert "700~1,200자" in prompt


def test_기관_비교_프롬프트에_전체_비교_표가_들어가지_않는다():
    """비교 결과가 100줄이어도 프롬프트에는 상위 종목만 들어가야 합니다."""
    left = [
        {
            "issuer_name": f"COMPANY {index:03d}",
            "class_title": "COM",
            "cusip": f"{index:09d}",
            "reported_value": 1000 - index,
            "shares": 100,
            "share_type": "SH",
            "put_call": "",
        }
        for index in range(60)
    ]
    right = [
        {**row, "cusip": f"{900 + index:09d}", "issuer_name": f"OTHER {index:03d}"}
        for index, row in enumerate(left[:40])
    ]

    comparison = compare_institution_portfolios(left, right)
    payload = build_institution_comparison_briefing_payload(comparison)
    prompt = build_institution_comparison_briefing_prompt(payload)

    assert len(comparison) == 100

    # 종목 한 줄에는 CUSIP이 한 번 나옵니다. 다섯 목록 * 최대 10개가 상한입니다.
    assert prompt.count("(CUSIP ") <= payload["top_n"] * 5

    # 어느 상위 목록에도 들지 않는 중간 순위 종목은 프롬프트에 없어야 합니다.
    assert "COMPANY 030" not in prompt

    # 내부 식별용 값과 공시 원문 관련 열 이름도 들어가지 않아야 합니다.
    for forbidden in ("position_key", "informationTable", "left_reported_value"):
        assert forbidden not in prompt


def test_기관_비교_결과가_비어도_프롬프트를_만들_수_있다():
    payload = build_institution_comparison_briefing_payload(
        compare_institution_portfolios([], [])
    )

    prompt = build_institution_comparison_briefing_prompt(payload)

    assert "해당 종목 없음" in prompt
    assert "확인 불가" in prompt  # 기준일과 기관명이 없을 때의 표시
    assert "공통 보유 종목 수: 0개" in prompt


@pytest.mark.parametrize("payload", [None, {}, "문자열"])
def test_기관_비교_입력_데이터가_없어도_프롬프트를_만들_수_있다(payload):
    """입력 데이터가 비어 있어도 오류 없이 지시문만 담은 프롬프트를 만듭니다."""
    prompt = build_institution_comparison_briefing_prompt(payload)

    assert "## 1. 한눈에 보는 비교" in prompt
    assert "해당 종목 없음" in prompt


def test_기관_비교_프롬프트에_API_키가_들어가지_않는다(institution_payload):
    prompt = build_institution_comparison_briefing_prompt(institution_payload)

    assert FAKE_API_KEY not in prompt


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
