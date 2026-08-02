"""services/sec_client.py 테스트.

실제 SEC 서버에 요청을 보내지 않습니다.
requests.get을 가짜(mock)로 바꿔치기하고, 예시 JSON으로만 검증합니다.
"""

from unittest.mock import patch

import pytest
import requests

from services.sec_client import (
    SUBMISSIONS_URL,
    InformationTableNotFoundError,
    SecApiError,
    build_filing_index_url,
    extract_recent_13f_filings,
    fetch_submissions,
    find_information_table_url,
    format_accession_number,
    get_13f_holdings,
    get_recent_13f_filings,
    normalize_accession_number,
    normalize_cik,
    parse_information_table,
)

# 테스트용 User-Agent. 실제 개인 이메일은 쓰지 않습니다.
TEST_USER_AGENT = "13F Insight Lab Test (test@example.com)"

# SEC submissions API 응답을 흉내 낸 예시 JSON.
# filings.recent는 '열 단위 배열' 구조라서, 같은 순번의 값들이 한 건의 공시를 이룹니다.
SAMPLE_SUBMISSIONS = {
    "cik": "1067983",
    "name": "BERKSHIRE HATHAWAY INC",
    "filings": {
        "recent": {
            "form": ["13F-HR", "8-K", "13F-HR/A", "13F-HR", "13F-HR", "4"],
            "accessionNumber": [
                "0000950123-25-008888",  # 13F-HR (가장 최신)
                "0000950123-25-007777",  # 8-K
                "0000950123-25-006666",  # 13F-HR/A (정정 공시)
                "0000950123-25-005555",  # 13F-HR (두 번째)
                "0000950123-24-004444",  # 13F-HR (세 번째)
                "0000950123-24-003333",  # 4
            ],
            "filingDate": [
                "2025-08-14",
                "2025-08-05",
                "2025-05-20",
                "2025-05-15",
                "2024-11-14",
                "2024-11-01",
            ],
            "reportDate": [
                "2025-06-30",
                "",
                "2025-03-31",
                "2025-03-31",
                "2024-09-30",
                "",
            ],
            "primaryDocument": [
                "xslForm13F_X02/primary_doc.xml",
                "form8k.htm",
                "xslForm13F_X02/primary_doc_a.xml",
                "xslForm13F_X02/primary_doc.xml",
                "xslForm13F_X02/primary_doc.xml",
                "form4.xml",
            ],
        }
    },
}


class FakeResponse:
    """requests의 응답 객체를 흉내 낸 테스트용 객체."""

    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Error", response=self
            )

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# 1. CIK 변환
# ---------------------------------------------------------------------------


def test_normalize_cik_pads_to_10_digits():
    """1067983이 0001067983으로 변환되는지 확인합니다."""
    assert normalize_cik(1067983) == "0001067983"
    assert normalize_cik("1067983") == "0001067983"


def test_normalize_cik_keeps_already_padded_value():
    """이미 10자리인 값은 그대로 유지되는지 확인합니다."""
    assert normalize_cik("0001067983") == "0001067983"


def test_normalize_cik_accepts_prefix_and_whitespace():
    """앞뒤 공백이나 'CIK' 접두어가 있어도 처리되는지 확인합니다."""
    assert normalize_cik("  CIK1067983 ") == "0001067983"


@pytest.mark.parametrize("bad_cik", ["", "abc123", "12345678901"])
def test_normalize_cik_rejects_invalid_value(bad_cik):
    """숫자가 아니거나 10자리를 넘는 값은 오류를 냅니다."""
    with pytest.raises(ValueError):
        normalize_cik(bad_cik)


# ---------------------------------------------------------------------------
# 2. 13F-HR 필터링 / 3. 최신 2건
# ---------------------------------------------------------------------------


def test_extract_only_13f_hr_forms():
    """form이 정확히 13F-HR인 항목만 골라내는지 확인합니다."""
    filings = extract_recent_13f_filings(SAMPLE_SUBMISSIONS, limit=None)

    # 예시 JSON에는 13F-HR이 3건, 정정 공시(13F-HR/A)가 1건 들어 있습니다.
    assert len(filings) == 3

    accession_numbers = [item["accession_number"] for item in filings]
    # 8-K, 4, 13F-HR/A 는 포함되지 않아야 합니다.
    assert "0000950123-25-007777" not in accession_numbers  # 8-K
    assert "0000950123-24-003333" not in accession_numbers  # 4
    assert "0000950123-25-006666" not in accession_numbers  # 13F-HR/A


def test_extract_returns_latest_two_filings():
    """최신순으로 최대 2건만 반환하는지 확인합니다."""
    filings = extract_recent_13f_filings(SAMPLE_SUBMISSIONS, limit=2)

    assert len(filings) == 2
    assert [item["filing_date"] for item in filings] == ["2025-08-14", "2025-05-15"]
    assert [item["accession_number"] for item in filings] == [
        "0000950123-25-008888",
        "0000950123-25-005555",
    ]


def test_extract_sorts_unordered_input_by_newest_first():
    """입력 순서가 뒤섞여 있어도 제출일 기준 최신순으로 정렬되는지 확인합니다."""
    unordered = {
        "filings": {
            "recent": {
                "form": ["13F-HR", "13F-HR", "13F-HR"],
                "accessionNumber": ["old", "newest", "middle"],
                "filingDate": ["2024-02-14", "2025-08-14", "2025-02-14"],
                "reportDate": ["2023-12-31", "2025-06-30", "2024-12-31"],
                "primaryDocument": ["a.xml", "b.xml", "c.xml"],
            }
        }
    }

    filings = extract_recent_13f_filings(unordered, limit=2)

    assert [item["accession_number"] for item in filings] == ["newest", "middle"]


def test_extract_returns_expected_keys():
    """반환 항목에 필요한 4개 키가 모두 담겨 있는지 확인합니다."""
    filings = extract_recent_13f_filings(SAMPLE_SUBMISSIONS, limit=1)

    assert filings[0] == {
        "accession_number": "0000950123-25-008888",
        "filing_date": "2025-08-14",
        "report_date": "2025-06-30",
        "primary_document": "xslForm13F_X02/primary_doc.xml",
    }


def test_extract_returns_empty_list_when_no_13f():
    """13F-HR 공시가 없으면 빈 목록을 돌려줍니다."""
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K"],
                "accessionNumber": ["a", "b"],
                "filingDate": ["2025-01-01", "2025-01-02"],
                "reportDate": ["", ""],
                "primaryDocument": ["a.htm", "b.htm"],
            }
        }
    }

    assert extract_recent_13f_filings(submissions) == []


def test_extract_handles_missing_filings_key():
    """예상과 다른 응답(filings 없음)에도 오류 없이 빈 목록을 돌려줍니다."""
    assert extract_recent_13f_filings({}) == []


# ---------------------------------------------------------------------------
# 4. HTTP 오류 처리 (네트워크 요청은 mock 처리)
# ---------------------------------------------------------------------------


def test_fetch_submissions_calls_expected_url_and_header():
    """올바른 주소와 User-Agent 헤더, 30초 timeout으로 요청하는지 확인합니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(json_data=SAMPLE_SUBMISSIONS)

        result = fetch_submissions(1067983, user_agent=TEST_USER_AGENT)

    assert result == SAMPLE_SUBMISSIONS

    called_url = mock_get.call_args.args[0]
    called_kwargs = mock_get.call_args.kwargs
    assert called_url == SUBMISSIONS_URL.format(cik="0001067983")
    assert called_url == "https://data.sec.gov/submissions/CIK0001067983.json"
    assert called_kwargs["headers"]["User-Agent"] == TEST_USER_AGENT
    assert called_kwargs["timeout"] == 30


@pytest.mark.parametrize("status_code", [403, 404, 500])
def test_fetch_submissions_raises_friendly_error_on_http_error(status_code):
    """HTTP 오류가 나면 이해하기 쉬운 SecApiError로 바꿔 알려 줍니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code=status_code)

        with pytest.raises(SecApiError) as error_info:
            fetch_submissions("0001067983", user_agent=TEST_USER_AGENT)

    message = str(error_info.value)
    assert "SEC 공시 목록을 불러오지 못했습니다" in message
    assert str(status_code) in message
    assert "0001067983" in message


def test_fetch_submissions_raises_on_timeout():
    """응답이 늦어 timeout이 발생하면 SecApiError로 알려 줍니다."""
    with patch("services.sec_client.requests.get", side_effect=requests.Timeout()):
        with pytest.raises(SecApiError) as error_info:
            fetch_submissions("0001067983", user_agent=TEST_USER_AGENT)

    assert "30초" in str(error_info.value)


def test_fetch_submissions_raises_on_connection_error():
    """연결 자체가 안 되면 SecApiError로 알려 줍니다."""
    with patch(
        "services.sec_client.requests.get", side_effect=requests.ConnectionError()
    ):
        with pytest.raises(SecApiError) as error_info:
            fetch_submissions("0001067983", user_agent=TEST_USER_AGENT)

    assert "연결하지 못했습니다" in str(error_info.value)


def test_fetch_submissions_raises_on_broken_json():
    """JSON 형식이 깨져 있으면 SecApiError로 알려 줍니다."""

    class BrokenJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("Expecting value")

    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = BrokenJsonResponse()

        with pytest.raises(SecApiError) as error_info:
            fetch_submissions("0001067983", user_agent=TEST_USER_AGENT)

    assert "이해할 수 없습니다" in str(error_info.value)


def test_fetch_submissions_requires_user_agent():
    """User-Agent가 비어 있으면 요청하지 않고 미리 오류를 냅니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        with pytest.raises(ValueError):
            fetch_submissions("0001067983", user_agent="   ")

    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# 통합: 조회 + 필터링을 묶은 대표 함수
# ---------------------------------------------------------------------------


def test_get_recent_13f_filings_returns_two_latest():
    """대표 함수가 최신 13F-HR 2건을 돌려주는지 확인합니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(json_data=SAMPLE_SUBMISSIONS)

        filings = get_recent_13f_filings(1067983, user_agent=TEST_USER_AGENT)

    assert len(filings) == 2
    assert filings[0]["report_date"] == "2025-06-30"
    assert filings[1]["report_date"] == "2025-03-31"
    mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# 5. 보유 종목 조회 - 예시 데이터
# ---------------------------------------------------------------------------

# 테스트에 사용할 공시 번호와 그에 대응하는 폴더 이름(하이픈 제거).
TEST_ACCESSION_NUMBER = "0000950123-25-008888"
TEST_ACCESSION_DIGITS = "000095012325008888"
TEST_FILING_DIRECTORY = (
    f"https://www.sec.gov/Archives/edgar/data/0001067983/{TEST_ACCESSION_DIGITS}"
)
TEST_INDEX_URL = f"{TEST_FILING_DIRECTORY}/{TEST_ACCESSION_NUMBER}-index.htm"

# 실제 SEC index 페이지의 링크는 0을 채우지 않은 CIK 경로를 씁니다.
# 아래는 예시 HTML의 링크에서 xsl 경로를 뺐을 때 나와야 하는 원본 XML 주소입니다.
EXPECTED_INFORMATION_TABLE_URL = (
    f"https://www.sec.gov/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}"
    "/form13fInfoTable.xml"
)

# SEC의 filing index 페이지를 흉내 낸 예시 HTML.
# 실제 페이지처럼 문서 유형(Type) 칸이 있고, 보유 종목 파일 링크는 사람이 보기 좋게
# 변환된 'xslForm13F_X01/' 경로를 가리킵니다.
SAMPLE_INDEX_HTML = f"""
<html><body>
  <p>Document Format Files</p>
  <table class="tableFile" summary="Document Format Files">
    <tr>
      <th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th>
    </tr>
    <tr>
      <td>1</td>
      <td>primary_doc.xml</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}/xslForm13F_X02/primary_doc.xml">primary_doc.xml</a></td>
      <td>13F-HR</td>
      <td>5678</td>
    </tr>
    <tr>
      <td>2</td>
      <td>form13fInfoTable.xml</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}/xslForm13F_X01/form13fInfoTable.xml">form13fInfoTable.xml</a></td>
      <td>INFORMATION TABLE</td>
      <td>123456</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Complete submission text file</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_NUMBER}.txt">{TEST_ACCESSION_NUMBER}.txt</a></td>
      <td>&nbsp;</td>
      <td>234567</td>
    </tr>
  </table>
</body></html>
"""

# 보유 종목 파일이 HTML과 XML 두 가지로 올라온 경우. HTML이 표에서 먼저 나옵니다.
SAMPLE_INDEX_HTML_WITH_TWO_INFORMATION_TABLES = f"""
<html><body>
  <table class="tableFile">
    <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
    <tr>
      <td>1</td><td>info table (사람이 보기용)</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}/infotable.html">infotable.html</a></td>
      <td>INFORMATION TABLE</td>
    </tr>
    <tr>
      <td>2</td><td>info table (원본)</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}/infotable.xml">infotable.xml</a></td>
      <td>INFORMATION TABLE</td>
    </tr>
  </table>
</body></html>
"""

# INFORMATION TABLE 줄이 아예 없는 index 페이지.
SAMPLE_INDEX_HTML_WITHOUT_INFORMATION_TABLE = f"""
<html><body>
  <table class="tableFile">
    <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
    <tr>
      <td>1</td><td>primary_doc.xml</td>
      <td><a href="/Archives/edgar/data/1067983/{TEST_ACCESSION_DIGITS}/primary_doc.xml">primary_doc.xml</a></td>
      <td>13F-HR</td>
    </tr>
  </table>
</body></html>
"""

# namespace가 없는 보유 종목 XML 예시.
# 세 번째 종목은 일부러 putCall을 빼고 value를 비워, 값이 없어도 파싱이 계속되는지
# 확인하기 위한 것입니다.
SAMPLE_INFORMATION_TABLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<informationTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>66643000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>300000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
    <votingAuthority>
      <Sole>300000000</Sole><Shared>0</Shared><None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>COCA COLA CO</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>191216100</cusip>
    <value>1,234,567</value>
    <shrsOrPrnAmt>
      <sshPrnamt>400,000,000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall>Call</putCall>
  </infoTable>
  <infoTable>
    <nameOfIssuer>KRAFT HEINZ CO</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>500754106</cusip>
    <value></value>
    <shrsOrPrnAmt>
      <sshPrnamt>325634818</sshPrnamt>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""

# namespace가 붙은 보유 종목 XML 예시.
# 접두어(ns1:)와 기본 namespace가 섞여 있어도 파싱되어야 합니다.
SAMPLE_INFORMATION_TABLE_XML_WITH_NAMESPACE = """<?xml version="1.0" encoding="UTF-8"?>
<ns1:informationTable
    xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <ns1:infoTable>
    <ns1:nameOfIssuer>BANK OF AMERICA CORP</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>060505104</ns1:cusip>
    <ns1:value>41074000</ns1:value>
    <ns1:shrsOrPrnAmt>
      <ns1:sshPrnamt>1032852006</ns1:sshPrnamt>
      <ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
    </ns1:shrsOrPrnAmt>
    <ns1:putCall>Put</ns1:putCall>
  </ns1:infoTable>
</ns1:informationTable>
"""


# ---------------------------------------------------------------------------
# 5-1. filing index 주소 만들기 (하이픈 제거)
# ---------------------------------------------------------------------------


def test_normalize_accession_number_removes_hyphens():
    """하이픈이 제거된 18자리 숫자가 되는지 확인합니다."""
    assert normalize_accession_number(TEST_ACCESSION_NUMBER) == TEST_ACCESSION_DIGITS
    # 이미 하이픈이 없는 값은 그대로 유지됩니다.
    assert normalize_accession_number(TEST_ACCESSION_DIGITS) == TEST_ACCESSION_DIGITS


def test_format_accession_number_restores_hyphens():
    """하이픈이 있는 원래 표기로 되돌리는지 확인합니다."""
    assert format_accession_number(TEST_ACCESSION_DIGITS) == TEST_ACCESSION_NUMBER


@pytest.mark.parametrize("bad_value", ["", "0000950123-25-88", "abcd-ef-ghijkl"])
def test_normalize_accession_number_rejects_invalid_value(bad_value):
    """숫자가 아니거나 자릿수가 맞지 않으면 오류를 냅니다."""
    with pytest.raises(ValueError):
        normalize_accession_number(bad_value)


def test_build_filing_index_url_strips_hyphens_from_folder():
    """폴더 이름에는 하이픈이 없고, 파일 이름에는 하이픈이 남는지 확인합니다."""
    url = build_filing_index_url("0001067983", TEST_ACCESSION_NUMBER)

    assert url == TEST_INDEX_URL
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/0001067983/"
        "000095012325008888/0000950123-25-008888-index.htm"
    )
    # 폴더 이름 부분에는 하이픈이 남아 있지 않아야 합니다.
    assert f"/{TEST_ACCESSION_DIGITS}/" in url


def test_build_filing_index_url_accepts_short_cik_and_plain_accession():
    """CIK가 짧거나 accession에 하이픈이 없어도 같은 주소를 만드는지 확인합니다."""
    assert (
        build_filing_index_url(1067983, TEST_ACCESSION_DIGITS)
        == build_filing_index_url("0001067983", TEST_ACCESSION_NUMBER)
    )


# ---------------------------------------------------------------------------
# 5-2. filing index에서 INFORMATION TABLE 찾기
# ---------------------------------------------------------------------------


def test_find_information_table_url_finds_xml_link():
    """문서 유형이 INFORMATION TABLE인 줄의 XML 링크를 찾는지 확인합니다."""
    url = find_information_table_url(SAMPLE_INDEX_HTML, base_url=TEST_INDEX_URL)

    assert url == EXPECTED_INFORMATION_TABLE_URL
    assert url.endswith("form13fInfoTable.xml")
    # 사람이 보기용으로 변환된 'xsl...' 경로는 빠져야 합니다(원본 XML이 필요).
    assert "xsl" not in url
    # 상대 주소가 아니라 전체 주소여야 합니다.
    assert url.startswith("https://www.sec.gov/")


def test_find_information_table_url_prefers_xml_over_html():
    """INFORMATION TABLE 파일이 여러 개면 XML을 먼저 고르는지 확인합니다."""
    url = find_information_table_url(
        SAMPLE_INDEX_HTML_WITH_TWO_INFORMATION_TABLES, base_url=TEST_INDEX_URL
    )

    assert url.endswith("infotable.xml")


def test_find_information_table_url_raises_when_missing():
    """INFORMATION TABLE이 없으면 이해하기 쉬운 예외를 냅니다."""
    with pytest.raises(InformationTableNotFoundError) as error_info:
        find_information_table_url(
            SAMPLE_INDEX_HTML_WITHOUT_INFORMATION_TABLE, base_url=TEST_INDEX_URL
        )

    message = str(error_info.value)
    assert "보유 종목 목록" in message
    assert TEST_INDEX_URL in message
    # SEC 관련 오류를 한꺼번에 처리하는 코드에서도 잡혀야 합니다.
    assert isinstance(error_info.value, SecApiError)


def test_find_information_table_url_raises_on_empty_index():
    """index 내용이 비어 있어도 오류로 멈추지 않고 친절한 예외를 냅니다."""
    with pytest.raises(InformationTableNotFoundError):
        find_information_table_url("", base_url=TEST_INDEX_URL)


# ---------------------------------------------------------------------------
# 5-3. 보유 종목 XML 파싱
# ---------------------------------------------------------------------------


def test_parse_information_table_reads_expected_fields():
    """종목명, CUSIP, 평가금액, 보유수량이 올바르게 파싱되는지 확인합니다."""
    holdings = parse_information_table(SAMPLE_INFORMATION_TABLE_XML)

    assert len(holdings) == 3
    assert holdings[0] == {
        "issuer_name": "APPLE INC",
        "class_title": "COM",
        "cusip": "037833100",
        "value_thousands": 66643000,
        "shares": 300000000,
        "share_type": "SH",
        "put_call": "",
    }


def test_parse_information_table_converts_numbers_with_commas():
    """쉼표가 들어간 숫자도 숫자형으로 바뀌는지 확인합니다."""
    holdings = parse_information_table(SAMPLE_INFORMATION_TABLE_XML)
    coca_cola = holdings[1]

    assert coca_cola["issuer_name"] == "COCA COLA CO"
    assert coca_cola["value_thousands"] == 1234567
    assert coca_cola["shares"] == 400000000
    assert isinstance(coca_cola["value_thousands"], int)
    assert coca_cola["put_call"] == "Call"


def test_parse_information_table_survives_missing_optional_fields():
    """빈 값이나 없는 선택 항목이 있어도 파싱이 중단되지 않는지 확인합니다."""
    holdings = parse_information_table(SAMPLE_INFORMATION_TABLE_XML)
    kraft = holdings[2]

    assert kraft["issuer_name"] == "KRAFT HEINZ CO"
    assert kraft["shares"] == 325634818
    # value가 비어 있으면 숫자로 바꿀 수 없으므로 None입니다.
    assert kraft["value_thousands"] is None
    # 없는 선택 항목은 빈 문자열로 채워집니다.
    assert kraft["share_type"] == ""
    assert kraft["put_call"] == ""


def test_parse_information_table_handles_namespace():
    """XML namespace가 있어도 같은 결과가 나오는지 확인합니다."""
    holdings = parse_information_table(SAMPLE_INFORMATION_TABLE_XML_WITH_NAMESPACE)

    assert len(holdings) == 1
    assert holdings[0] == {
        "issuer_name": "BANK OF AMERICA CORP",
        "class_title": "COM",
        "cusip": "060505104",
        "value_thousands": 41074000,
        "shares": 1032852006,
        "share_type": "SH",
        "put_call": "Put",
    }


def test_parse_information_table_returns_empty_list_when_no_holdings():
    """종목이 없는 XML은 빈 목록을 돌려줍니다."""
    assert parse_information_table("<informationTable></informationTable>") == []


def test_parse_information_table_raises_on_broken_xml():
    """XML 형식이 깨져 있으면 이해하기 쉬운 SecApiError로 알려 줍니다."""
    with pytest.raises(SecApiError) as error_info:
        parse_information_table("<informationTable><infoTable>")

    assert "형식이 올바르지 않아" in str(error_info.value)


# ---------------------------------------------------------------------------
# 5-4. 통합: 주소 만들기 + index 조회 + XML 파싱 (네트워크는 mock)
# ---------------------------------------------------------------------------


def test_get_13f_holdings_requests_index_then_xml():
    """index를 먼저 받고, 거기서 찾은 XML을 이어서 받는지 확인합니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.side_effect = [
            FakeResponse(text=SAMPLE_INDEX_HTML),
            FakeResponse(text=SAMPLE_INFORMATION_TABLE_XML),
        ]

        holdings = get_13f_holdings(
            1067983,
            TEST_ACCESSION_NUMBER,
            user_agent=TEST_USER_AGENT,
        )

    assert len(holdings) == 3
    assert holdings[0]["issuer_name"] == "APPLE INC"
    assert holdings[0]["cusip"] == "037833100"

    # 요청은 두 번(index, XML)만 나가야 합니다.
    assert mock_get.call_count == 2

    index_call, xml_call = mock_get.call_args_list
    assert index_call.args[0] == TEST_INDEX_URL
    assert xml_call.args[0].endswith("form13fInfoTable.xml")
    assert "xsl" not in xml_call.args[0]

    # 두 요청 모두 전달받은 User-Agent와 30초 timeout을 사용해야 합니다.
    for call in (index_call, xml_call):
        assert call.kwargs["headers"]["User-Agent"] == TEST_USER_AGENT
        assert call.kwargs["timeout"] == 30


@pytest.mark.parametrize("status_code", [403, 404, 500])
def test_get_13f_holdings_raises_friendly_error_on_http_error(status_code):
    """HTTP 오류가 나면 이해하기 쉬운 SecApiError로 바꿔 알려 줍니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code=status_code)

        with pytest.raises(SecApiError) as error_info:
            get_13f_holdings(
                1067983,
                TEST_ACCESSION_NUMBER,
                user_agent=TEST_USER_AGENT,
            )

    message = str(error_info.value)
    assert "공시 문서 목록을 불러오지 못했습니다" in message
    assert str(status_code) in message


def test_get_13f_holdings_raises_friendly_error_when_xml_download_fails():
    """XML을 받는 두 번째 요청이 실패해도 친절한 오류로 바꿔 줍니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.side_effect = [
            FakeResponse(text=SAMPLE_INDEX_HTML),
            FakeResponse(status_code=404),
        ]

        with pytest.raises(SecApiError) as error_info:
            get_13f_holdings(
                1067983,
                TEST_ACCESSION_NUMBER,
                user_agent=TEST_USER_AGENT,
            )

    assert "보유 종목 목록을 불러오지 못했습니다" in str(error_info.value)


def test_get_13f_holdings_raises_on_timeout():
    """응답이 늦어 timeout이 발생하면 SecApiError로 알려 줍니다."""
    with patch("services.sec_client.requests.get", side_effect=requests.Timeout()):
        with pytest.raises(SecApiError) as error_info:
            get_13f_holdings(
                1067983,
                TEST_ACCESSION_NUMBER,
                user_agent=TEST_USER_AGENT,
            )

    assert "30초" in str(error_info.value)


def test_get_13f_holdings_requires_user_agent():
    """User-Agent가 비어 있으면 요청하지 않고 미리 오류를 냅니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        with pytest.raises(ValueError):
            get_13f_holdings(1067983, TEST_ACCESSION_NUMBER, user_agent="   ")

    mock_get.assert_not_called()


def test_get_13f_holdings_raises_when_information_table_missing():
    """index에 INFORMATION TABLE이 없으면 전용 예외를 냅니다."""
    with patch("services.sec_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(
            text=SAMPLE_INDEX_HTML_WITHOUT_INFORMATION_TABLE
        )

        with pytest.raises(InformationTableNotFoundError):
            get_13f_holdings(
                1067983,
                TEST_ACCESSION_NUMBER,
                user_agent=TEST_USER_AGENT,
            )

    # XML을 받으러 두 번째 요청을 보내지 않아야 합니다.
    assert mock_get.call_count == 1
