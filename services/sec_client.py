"""SEC 13F 공시 데이터 수집 모듈.

역할:
    미국 증권거래위원회(SEC) EDGAR 시스템에서 특정 기관투자자(CIK 기준)의
    13F-HR 공시를 찾아 보유 종목 목록(종목명, CUSIP, 평가금액, 보유 주식 수 등)을
    가져오는 책임을 맡습니다.

현재 구현된 기능:
    - CIK를 10자리 형식으로 맞추기 (normalize_cik)
    - SEC submissions API 호출 (fetch_submissions)
    - 정기 13F-HR 공시 목록 골라내기 (extract_recent_13f_filings)

포함 예정 기능:
    - 특정 분기 공시의 보유 종목 상세(information table) 파싱
    - 조회 결과를 pandas DataFrame 형태로 변환

주의:
    - SEC EDGAR는 요청 시 User-Agent 헤더(담당자 이메일)를 요구합니다.
      이메일 등 식별 정보는 코드에 직접 쓰지 않고 st.secrets 등 외부 설정에서
      읽어와 함수 인자로 전달합니다.
    - SEC의 요청 빈도 제한(초당 10회 이하)을 지켜야 합니다.
    - SEC submissions API는 별도의 API 키가 필요하지 않습니다.
"""

from __future__ import annotations

import requests

# SEC submissions API 주소 틀. {cik} 자리에 10자리로 맞춘 CIK가 들어갑니다.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC 권장 사항에 따른 기본 요청 제한 시간(초).
DEFAULT_TIMEOUT = 30

# 이번 단계에서 찾는 공시 종류. 정정 공시(13F-HR/A)는 포함하지 않습니다.
FORM_13F_HR = "13F-HR"

# CIK 자릿수. SEC는 앞에 0을 채운 10자리 형식을 사용합니다.
CIK_LENGTH = 10


class SecApiError(RuntimeError):
    """SEC EDGAR 호출이 실패했을 때 발생하는 예외.

    화면에 그대로 보여 줄 수 있도록, 사람이 읽기 쉬운 한국어 메시지를 담습니다.
    """


def normalize_cik(cik: str | int) -> str:
    """CIK를 SEC가 요구하는 10자리 문자열로 맞춥니다.

    예: 1067983 -> "0001067983"

    Args:
        cik: 숫자 또는 문자열 형태의 CIK. "CIK0001067983" 처럼 접두어가 붙어 있거나
            앞뒤 공백이 있어도 처리합니다.

    Returns:
        앞에 0을 채운 10자리 CIK 문자열.

    Raises:
        ValueError: 숫자가 아닌 값이 섞여 있거나, 10자리를 넘는 경우.
    """
    text = str(cik).strip()

    # "CIK0001067983" 처럼 접두어가 붙은 값도 받아 줍니다.
    if text.upper().startswith("CIK"):
        text = text[3:]

    if not text.isdigit():
        raise ValueError(f"CIK는 숫자로만 이루어져야 합니다: {cik!r}")

    if len(text) > CIK_LENGTH:
        raise ValueError(f"CIK가 {CIK_LENGTH}자리를 넘습니다: {cik!r}")

    return text.zfill(CIK_LENGTH)


def fetch_submissions(
    cik: str | int,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """SEC submissions API를 호출해 공시 이력 원본 JSON을 가져옵니다.

    Args:
        cik: 조회할 기관의 CIK. 10자리보다 짧으면 자동으로 0을 채웁니다.
        user_agent: SEC가 요구하는 User-Agent 값(회사명과 연락처).
            코드에 직접 쓰지 않고 st.secrets 등에서 읽어와 전달합니다.
        timeout: 응답을 기다리는 최대 시간(초).

    Returns:
        SEC가 돌려준 JSON을 파이썬 dict로 변환한 값.

    Raises:
        ValueError: user_agent가 비어 있는 경우.
        SecApiError: 통신 실패, HTTP 오류, JSON 형식 오류가 발생한 경우.
    """
    if not user_agent or not user_agent.strip():
        raise ValueError(
            "SEC EDGAR는 연락처가 담긴 User-Agent 값을 요구합니다. "
            "user_agent를 반드시 전달해 주세요."
        )

    normalized_cik = normalize_cik(cik)
    url = SUBMISSIONS_URL.format(cik=normalized_cik)
    headers = {
        "User-Agent": user_agent.strip(),
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", None)
        raise SecApiError(
            f"SEC 공시 목록을 불러오지 못했습니다 (CIK {normalized_cik}, "
            f"HTTP 상태 코드 {status_code}). "
            "CIK가 올바른지, User-Agent에 연락처가 들어 있는지 확인해 주세요."
        ) from error
    except requests.Timeout as error:
        raise SecApiError(
            f"SEC 응답이 {timeout}초 안에 오지 않았습니다 (CIK {normalized_cik}). "
            "잠시 후 다시 시도해 주세요."
        ) from error
    except requests.RequestException as error:
        raise SecApiError(
            f"SEC에 연결하지 못했습니다 (CIK {normalized_cik}). "
            "인터넷 연결 상태를 확인해 주세요."
        ) from error

    try:
        return response.json()
    except ValueError as error:
        raise SecApiError(
            f"SEC 응답을 이해할 수 없습니다 (CIK {normalized_cik}). "
            "SEC 서버가 일시적으로 불안정할 수 있습니다."
        ) from error


def extract_recent_13f_filings(submissions: dict, limit: int = 2) -> list[dict]:
    """submissions JSON에서 정기 13F-HR 공시만 골라 최신순으로 반환합니다.

    SEC의 filings.recent는 항목별 목록이 아니라 '열 단위 배열' 구조입니다.
    즉 form, accessionNumber, filingDate 등이 각각 같은 길이의 배열이고,
    같은 순번(index)에 있는 값들이 하나의 공시를 이룹니다.
    이 함수는 그 구조를 다루기 쉬운 dict 목록으로 바꿔 줍니다.

    Args:
        submissions: fetch_submissions가 돌려준 JSON dict.
        limit: 최대 몇 건을 돌려줄지. 기본값 2건.

    Returns:
        최신순으로 정렬된 공시 목록. 각 항목은 accession_number, filing_date,
        report_date, primary_document 키를 가집니다. 조건에 맞는 공시가 없으면
        빈 목록을 돌려줍니다.
    """
    recent = (submissions or {}).get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])

    filings: list[dict] = []
    for index, form in enumerate(forms):
        # 정확히 "13F-HR"인 것만. 정정 공시(13F-HR/A)는 제외합니다.
        if form != FORM_13F_HR:
            continue

        filings.append(
            {
                "accession_number": _value_at(accession_numbers, index),
                "filing_date": _value_at(filing_dates, index),
                "report_date": _value_at(report_dates, index),
                "primary_document": _value_at(primary_documents, index),
            }
        )

    # SEC는 보통 최신순으로 내려주지만, 순서를 보장받기 위해 제출일 기준으로 다시 정렬합니다.
    filings.sort(key=lambda item: item["filing_date"] or "", reverse=True)

    if limit is not None and limit >= 0:
        return filings[:limit]
    return filings


def get_recent_13f_filings(
    cik: str | int,
    user_agent: str,
    limit: int = 2,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """CIK로 최근 정기 13F-HR 공시의 기본 정보를 조회합니다.

    화면(streamlit_app.py)에서 사용할 대표 함수입니다.

    Args:
        cik: 조회할 기관의 CIK. 예) Berkshire Hathaway는 "0001067983".
        user_agent: SEC가 요구하는 User-Agent 값(회사명과 연락처).
        limit: 최대 몇 건을 돌려줄지. 기본값 2건.
        timeout: 응답을 기다리는 최대 시간(초).

    Returns:
        최신순 공시 목록(accession_number, filing_date, report_date,
        primary_document).

    Raises:
        SecApiError: SEC 호출이 실패한 경우.
    """
    submissions = fetch_submissions(cik, user_agent=user_agent, timeout=timeout)
    return extract_recent_13f_filings(submissions, limit=limit)


def _value_at(values: list, index: int):
    """배열에서 index 위치의 값을 안전하게 꺼냅니다. 없으면 None을 돌려줍니다.

    SEC 응답에서 일부 열의 길이가 다를 때 오류로 멈추지 않게 하기 위한 보조 함수입니다.
    """
    if index < len(values):
        return values[index]
    return None
