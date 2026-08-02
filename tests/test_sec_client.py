"""services/sec_client.py 테스트.

실제 SEC 서버에 요청을 보내지 않습니다.
requests.get을 가짜(mock)로 바꿔치기하고, 예시 JSON으로만 검증합니다.
"""

from unittest.mock import patch

import pytest
import requests

from services.sec_client import (
    SUBMISSIONS_URL,
    SecApiError,
    extract_recent_13f_filings,
    fetch_submissions,
    get_recent_13f_filings,
    normalize_cik,
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

    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

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
