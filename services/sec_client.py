"""SEC 13F 공시 데이터 수집 모듈.

역할:
    미국 증권거래위원회(SEC) EDGAR 시스템에서 특정 기관투자자(CIK 기준)의
    13F-HR 공시를 찾아 보유 종목 목록(종목명, CUSIP, 평가금액, 보유 주식 수 등)을
    가져오는 책임을 맡습니다.

현재 구현된 기능:
    - CIK를 10자리 형식으로 맞추기 (normalize_cik)
    - SEC submissions API 호출 (fetch_submissions)
    - 정기 13F-HR 공시 목록 골라내기 (extract_recent_13f_filings)
    - 공시 문서 목록(filing index)에서 보유 종목 파일 찾기
      (build_filing_index_url, fetch_filing_index, find_information_table_url)
    - 보유 종목 상세(information table) XML 파싱 (parse_information_table)
    - 위 과정을 한 번에 수행하는 대표 함수 (get_13f_holdings)

포함 예정 기능:
    - 조회 결과를 pandas DataFrame 형태로 변환

주의:
    - SEC EDGAR는 요청 시 User-Agent 헤더(담당자 이메일)를 요구합니다.
      이메일 등 식별 정보는 코드에 직접 쓰지 않고 st.secrets 등 외부 설정에서
      읽어와 함수 인자로 전달합니다.
    - SEC의 요청 빈도 제한(초당 10회 이하)을 지켜야 합니다.
    - SEC submissions API는 별도의 API 키가 필요하지 않습니다.
"""

from __future__ import annotations

from urllib.parse import urljoin
from xml.etree import ElementTree

import requests
from lxml import html as lxml_html

# SEC submissions API 주소 틀. {cik} 자리에 10자리로 맞춘 CIK가 들어갑니다.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# 공시 원본 문서가 보관된 곳(EDGAR Archives)의 주소 틀.
# {accession} 에는 하이픈을 제거한 18자리, {accession_dashed} 에는 하이픈이 있는
# 원래 형태가 들어갑니다. 폴더 이름은 하이픈이 없고, index 파일 이름에는 하이픈이
# 있는 SEC의 규칙을 그대로 따른 것입니다.
FILING_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    "{accession_dashed}-index.htm"
)

# SEC 권장 사항에 따른 기본 요청 제한 시간(초).
DEFAULT_TIMEOUT = 30

# 이번 단계에서 찾는 공시 종류. 정정 공시(13F-HR/A)는 포함하지 않습니다.
FORM_13F_HR = "13F-HR"

# CIK 자릿수. SEC는 앞에 0을 채운 10자리 형식을 사용합니다.
CIK_LENGTH = 10

# accession number(공시 접수 번호) 자릿수. 하이픈을 뺀 숫자만 세었을 때의 길이입니다.
ACCESSION_NUMBER_LENGTH = 18

# filing index의 '문서 유형(Type)' 칸에서 찾을 값.
# 보유 종목이 담긴 파일은 이 유형으로 표시됩니다.
INFORMATION_TABLE_TYPE = "INFORMATION TABLE"


class SecApiError(RuntimeError):
    """SEC EDGAR 호출이 실패했을 때 발생하는 예외.

    화면에 그대로 보여 줄 수 있도록, 사람이 읽기 쉬운 한국어 메시지를 담습니다.
    """


class InformationTableNotFoundError(SecApiError):
    """공시 문서 목록에서 보유 종목 파일(INFORMATION TABLE)을 찾지 못했을 때의 예외.

    SecApiError를 상속하므로, SEC 관련 오류를 한꺼번에 처리하는 코드에서도 잡힙니다.
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
    clean_user_agent = _require_user_agent(user_agent)

    normalized_cik = normalize_cik(cik)
    url = SUBMISSIONS_URL.format(cik=normalized_cik)
    headers = {
        "User-Agent": clean_user_agent,
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


# ---------------------------------------------------------------------------
# 보유 종목(information table) 조회
# ---------------------------------------------------------------------------


def normalize_accession_number(accession_number: str | int) -> str:
    """accession number(공시 접수 번호)에서 하이픈과 공백을 제거합니다.

    SEC 보관소의 폴더 이름은 하이픈이 없는 18자리 숫자입니다.

    예: "0000950123-25-008888" -> "000095012325008888"

    Args:
        accession_number: 하이픈이 있어도 없어도 됩니다.

    Returns:
        숫자 18자리 문자열.

    Raises:
        ValueError: 숫자가 아닌 값이 섞여 있거나 18자리가 아닌 경우.
    """
    text = str(accession_number).strip().replace("-", "").replace(" ", "")

    if not text.isdigit():
        raise ValueError(
            f"accession number는 숫자와 하이픈으로만 이루어져야 합니다: "
            f"{accession_number!r}"
        )

    if len(text) != ACCESSION_NUMBER_LENGTH:
        raise ValueError(
            f"accession number는 하이픈을 뺀 숫자 {ACCESSION_NUMBER_LENGTH}자리여야 "
            f"합니다: {accession_number!r}"
        )

    return text


def format_accession_number(accession_number: str | int) -> str:
    """accession number를 하이픈이 들어간 원래 표기로 되돌립니다.

    SEC의 filing index 파일 이름은 하이픈이 있는 형태를 씁니다.

    예: "000095012325008888" -> "0000950123-25-008888"
    """
    digits = normalize_accession_number(accession_number)
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def build_filing_index_url(cik: str | int, accession_number: str | int) -> str:
    """공시 문서 목록(filing index) 페이지 주소를 만듭니다.

    폴더 이름에는 하이픈을 제거한 accession number를, 파일 이름에는 하이픈이 있는
    형태를 사용하는 SEC의 규칙을 따릅니다.

    Args:
        cik: 조회할 기관의 CIK.
        accession_number: 공시 접수 번호. 하이픈이 있어도 없어도 됩니다.

    Returns:
        filing index 페이지의 전체 주소.

    Raises:
        ValueError: CIK나 accession number 형식이 올바르지 않은 경우.
    """
    return FILING_INDEX_URL.format(
        cik=normalize_cik(cik),
        accession=normalize_accession_number(accession_number),
        accession_dashed=format_accession_number(accession_number),
    )


def fetch_filing_index(
    cik: str | int,
    accession_number: str | int,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """filing index 페이지의 내용(HTML 문자열)을 받아옵니다.

    Args:
        cik: 조회할 기관의 CIK.
        accession_number: 공시 접수 번호.
        user_agent: SEC가 요구하는 User-Agent 값(회사명과 연락처).
        timeout: 응답을 기다리는 최대 시간(초).

    Returns:
        index 페이지의 HTML 문자열.

    Raises:
        ValueError: user_agent가 비어 있거나 CIK/accession 형식이 잘못된 경우.
        SecApiError: 통신 실패나 HTTP 오류가 발생한 경우.
    """
    url = build_filing_index_url(cik, accession_number)
    return _fetch_text(
        url,
        user_agent=user_agent,
        timeout=timeout,
        accept="text/html",
        description="공시 문서 목록",
    )


def find_information_table_url(index_html: str, base_url: str) -> str:
    """filing index 내용에서 보유 종목 파일의 주소를 찾습니다.

    파일 이름을 미리 정해 두지 않고, index 표에서 '문서 유형(Type)' 칸이
    INFORMATION TABLE인 줄을 찾아 그 줄에 걸린 링크를 사용합니다.
    후보가 여러 개면 XML 파일을 먼저 고릅니다.

    Args:
        index_html: fetch_filing_index가 돌려준 HTML 문자열.
        base_url: 상대 주소를 절대 주소로 바꿀 때 기준이 되는 주소
            (보통 index 페이지 주소).

    Returns:
        보유 종목 파일의 전체 주소.

    Raises:
        InformationTableNotFoundError: INFORMATION TABLE 파일을 찾지 못한 경우.
    """
    candidates = _collect_information_table_links(index_html)

    if not candidates:
        raise InformationTableNotFoundError(
            "이 공시에서 보유 종목 목록(INFORMATION TABLE) 파일을 찾지 못했습니다. "
            f"문서 목록 주소를 직접 확인해 주세요: {base_url}"
        )

    # 같은 내용을 HTML로도 제공하는 경우가 있어, 기계가 읽기 쉬운 XML을 우선합니다.
    xml_candidates = [link for link in candidates if link.lower().endswith(".xml")]
    chosen = (xml_candidates or candidates)[0]

    return urljoin(base_url, chosen)


def parse_information_table(xml_text: str) -> list[dict]:
    """보유 종목 XML(information table)을 딕셔너리 목록으로 바꿉니다.

    XML namespace(문서마다 다를 수 있는 이름 공간)에 상관없이 동작하도록,
    태그 이름의 뒷부분만 비교합니다.

    값이 비어 있거나 선택 항목이 아예 없어도 전체 파싱이 멈추지 않습니다.
    글자 항목은 빈 문자열(""), 숫자 항목은 None으로 채워집니다.

    Args:
        xml_text: information table XML 문자열.

    Returns:
        종목별 딕셔너리 목록. 각 딕셔너리는 다음 키를 가집니다.
            issuer_name: 발행사명
            class_title: 증권 종류
            cusip: CUSIP(증권 고유 번호)
            value_thousands: 공시 평가금액
            shares: 보유수량
            share_type: 주식 수량 단위(SH=주식, PRN=원금)
            put_call: Put 또는 Call 정보. 일반 주식은 빈 문자열입니다.

    Raises:
        SecApiError: XML 형식이 깨져서 읽을 수 없는 경우.

    Note:
        value_thousands는 SEC 서식의 value 칸을 그대로 담습니다. SEC는 2023년
        중반에 이 칸의 단위를 '천 달러'에서 '달러'로 바꿨으므로, 화면에 표시할 때는
        공시 시점을 함께 고려해야 합니다.
    """
    try:
        root = ElementTree.fromstring((xml_text or "").strip())
    except ElementTree.ParseError as error:
        raise SecApiError(
            "보유 종목 파일(XML)의 형식이 올바르지 않아 읽을 수 없습니다. "
            "SEC 문서가 일시적으로 잘못 내려왔을 수 있으니 다시 시도해 주세요."
        ) from error

    holdings: list[dict] = []
    for element in root.iter():
        if _local_name(element.tag) != "infoTable":
            continue
        holdings.append(
            {
                "issuer_name": _descendant_text(element, "nameOfIssuer"),
                "class_title": _descendant_text(element, "titleOfClass"),
                "cusip": _descendant_text(element, "cusip"),
                "value_thousands": _to_number(_descendant_text(element, "value")),
                # 보유수량과 단위는 shrsOrPrnAmt 안에 들어 있지만, 하위 태그까지
                # 훑기 때문에 감싸는 태그가 없거나 달라도 찾아냅니다.
                "shares": _to_number(_descendant_text(element, "sshPrnamt")),
                "share_type": _descendant_text(element, "sshPrnamtType"),
                "put_call": _descendant_text(element, "putCall"),
            }
        )

    return holdings


def get_13f_holdings(
    cik: str | int,
    accession_number: str | int,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """특정 13F 공시의 보유 종목 목록을 조회합니다.

    순서:
        1) filing index 주소를 만들고 내용을 받아옵니다.
        2) 그 안에서 INFORMATION TABLE 파일 주소를 찾습니다.
        3) 그 파일을 받아 종목 정보로 파싱합니다.

    Args:
        cik: 조회할 기관의 CIK. 예) Berkshire Hathaway는 "0001067983".
        accession_number: 공시 접수 번호. get_recent_13f_filings 결과의
            accession_number를 그대로 넣으면 됩니다.
        user_agent: SEC가 요구하는 User-Agent 값(회사명과 연락처).
        timeout: 응답을 기다리는 최대 시간(초).

    Returns:
        종목별 딕셔너리 목록. 자세한 키 설명은 parse_information_table을 참고하세요.

    Raises:
        ValueError: user_agent가 비어 있거나 CIK/accession 형식이 잘못된 경우.
        InformationTableNotFoundError: 보유 종목 파일을 찾지 못한 경우.
        SecApiError: SEC 호출이 실패하거나 XML을 읽을 수 없는 경우.
    """
    index_url = build_filing_index_url(cik, accession_number)
    index_html = _fetch_text(
        index_url,
        user_agent=user_agent,
        timeout=timeout,
        accept="text/html",
        description="공시 문서 목록",
    )

    information_table_url = find_information_table_url(index_html, base_url=index_url)
    xml_text = _fetch_text(
        information_table_url,
        user_agent=user_agent,
        timeout=timeout,
        accept="application/xml",
        description="보유 종목 목록",
    )

    return parse_information_table(xml_text)


# ---------------------------------------------------------------------------
# 내부 보조 함수 (모듈 밖에서 직접 쓰지 않습니다)
# ---------------------------------------------------------------------------


def _value_at(values: list, index: int):
    """배열에서 index 위치의 값을 안전하게 꺼냅니다. 없으면 None을 돌려줍니다.

    SEC 응답에서 일부 열의 길이가 다를 때 오류로 멈추지 않게 하기 위한 보조 함수입니다.
    """
    if index < len(values):
        return values[index]
    return None


def _require_user_agent(user_agent: str) -> str:
    """User-Agent 값이 비어 있지 않은지 확인하고 앞뒤 공백을 정리해 돌려줍니다."""
    if not user_agent or not user_agent.strip():
        raise ValueError(
            "SEC EDGAR는 연락처가 담긴 User-Agent 값을 요구합니다. "
            "user_agent를 반드시 전달해 주세요."
        )
    return user_agent.strip()


def _fetch_text(
    url: str,
    user_agent: str,
    timeout: int,
    accept: str,
    description: str,
) -> str:
    """SEC에서 문서를 받아 문자열로 돌려줍니다. 오류는 SecApiError로 바꿉니다.

    Args:
        url: 받아올 문서의 전체 주소.
        user_agent: SEC가 요구하는 User-Agent 값.
        timeout: 응답을 기다리는 최대 시간(초).
        accept: Accept 헤더 값.
        description: 오류 메시지에 넣을 한국어 설명(예: "공시 문서 목록").
    """
    headers = {
        "User-Agent": _require_user_agent(user_agent),
        "Accept": accept,
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", None)
        raise SecApiError(
            f"{description}을 불러오지 못했습니다 (HTTP 상태 코드 {status_code}). "
            f"공시 번호가 올바른지 확인해 주세요. 요청한 주소: {url}"
        ) from error
    except requests.Timeout as error:
        raise SecApiError(
            f"{description}에 대한 SEC 응답이 {timeout}초 안에 오지 않았습니다. "
            "잠시 후 다시 시도해 주세요."
        ) from error
    except requests.RequestException as error:
        raise SecApiError(
            f"{description}을 받아오는 중 SEC에 연결하지 못했습니다. "
            "인터넷 연결 상태를 확인해 주세요."
        ) from error

    return response.text or ""


def _collect_information_table_links(index_html: str) -> list[str]:
    """index HTML의 표에서 INFORMATION TABLE 줄에 걸린 링크를 모두 모읍니다.

    문서 유형 칸이 INFORMATION TABLE인 줄(tr)을 찾고, 그 줄 안의 링크(a) 주소를
    가져옵니다. 읽을 수 없는 HTML이면 빈 목록을 돌려줍니다.
    """
    if not index_html or not index_html.strip():
        return []

    try:
        root = lxml_html.fromstring(index_html)
    except Exception:  # lxml은 상황에 따라 여러 종류의 예외를 냅니다.
        return []

    links: list[str] = []
    for row in root.iter("tr"):
        cell_texts = [_clean_text(cell.text_content()) for cell in row.iter("td")]
        if not any(text.upper() == INFORMATION_TABLE_TYPE for text in cell_texts):
            continue

        for anchor in row.iter("a"):
            href = (anchor.get("href") or "").strip()
            if href:
                links.append(_strip_xsl_path(href))

    return links


def _strip_xsl_path(href: str) -> str:
    """사람이 보기 좋게 변환된 주소를 원본 파일 주소로 바꿉니다.

    SEC의 index 페이지는 보유 종목 파일을 표 형태로 보여 주는 변환 주소
    (예: .../xslForm13F_X01/form13fInfoTable.xml)를 걸어 두는 경우가 많습니다.
    파싱에는 원본 XML이 필요하므로 'xsl...'로 시작하는 경로 조각을 떼어 냅니다.
    """
    parts = href.split("/")
    kept = [part for part in parts if not part.lower().startswith("xsl")]

    # 모든 조각이 사라지는 비정상적인 경우에는 원래 주소를 그대로 씁니다.
    if not any(part for part in kept):
        return href

    return "/".join(kept)


def _local_name(tag) -> str:
    """XML 태그 이름에서 namespace를 떼어 낸 뒷부분만 돌려줍니다.

    예: "{http://www.sec.gov/edgar/...}infoTable" -> "infoTable"

    주석(comment)처럼 태그 이름이 문자열이 아닌 요소는 빈 문자열로 처리합니다.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _descendant_text(element, name: str) -> str:
    """element 안에서 이름이 일치하는 첫 하위 태그의 글자를 돌려줍니다.

    namespace는 무시하고 이름 뒷부분만 비교합니다. 해당 태그가 없거나 내용이
    비어 있으면 빈 문자열을 돌려주므로, 선택 항목이 없어도 파싱이 멈추지 않습니다.
    """
    for child in element.iter():
        if child is element or _local_name(child.tag) != name:
            continue
        return _clean_text(child.text or "")
    return ""


def _to_number(text: str):
    """문자열을 숫자로 바꿉니다. 바꿀 수 없으면 None을 돌려줍니다.

    "1,234"처럼 쉼표가 들어간 값도 처리합니다. 소수점이 없으면 정수로,
    소수점이 있으면 실수로 돌려줍니다.
    """
    cleaned = (text or "").replace(",", "").replace(" ", "")
    if not cleaned:
        return None

    try:
        # 아주 큰 정수도 오차 없이 다루기 위해 정수 변환을 먼저 시도합니다.
        return int(cleaned)
    except ValueError:
        pass

    try:
        number = float(cleaned)
    except ValueError:
        return None

    return int(number) if number.is_integer() else number


def _clean_text(text: str) -> str:
    """앞뒤 공백과 줄바꿈, 줄바꿈 없는 공백(&nbsp;)을 정리합니다."""
    return " ".join((text or "").replace("\xa0", " ").split())
