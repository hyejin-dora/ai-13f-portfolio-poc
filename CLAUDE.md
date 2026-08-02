# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 참고하는 프로젝트 안내서입니다.

## 프로젝트 개요

SEC 13F 공시 데이터를 분석하고, 그 결과를 Gemini API가 자연어로 설명해 주는
Streamlit 웹 애플리케이션 PoC(개념 검증)입니다. 교육 목적이며 투자 자문이 아닙니다.

- 분석 대상: Berkshire Hathaway (CIK 0001067983)
- 사용자는 개발 비전공자입니다. 설명은 한국어로, 전문 용어는 풀어서 씁니다.

## 폴더 구조

```
streamlit_app.py            # 화면(UI) 진입점
requirements.txt            # 필요한 파이썬 라이브러리 목록
services/
  sec_client.py             # SEC EDGAR에서 13F 데이터 수집
  portfolio_analysis.py     # 보유 비중, 분기별 변화 등 계산
  llm_client.py             # Gemini API로 자연어 설명 생성
data/managers.csv           # 분석 대상 운용사 목록 (name, cik)
tests/                      # pytest 테스트 코드
```

화면(`streamlit_app.py`)과 로직(`services/`)을 분리합니다.
새로운 기능은 `services/`의 적절한 모듈에 추가하고, 화면에서는 호출만 합니다.

## 보안 규칙 (반드시 지킬 것)

- `.streamlit/secrets.toml`은 읽지도, 수정하지도, 화면에 출력하지도 않습니다.
- API 키, 이메일 주소 등 비밀 정보를 코드나 문서에 직접 쓰지 않습니다.
  필요할 때는 `st.secrets["..."]`로 읽어옵니다.
- 비밀 정보가 담긴 파일은 절대 커밋하지 않습니다 (`.gitignore`로 관리 중).

## 작업 방식

- 변경할 파일과 그 이유를 먼저 한국어로 설명한 뒤 작업합니다.
- Git commit / push는 사용자가 명시적으로 요청할 때만 실행합니다.
- 외부 API(SEC, Gemini) 호출은 해당 기능을 구현하는 단계에서만 수행합니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

테스트 실행:

```bash
pytest
```

## 참고 사항

- SEC EDGAR는 요청 시 연락처가 담긴 User-Agent 헤더를 요구하며,
  초당 10회 이하의 요청 빈도 제한이 있습니다.
- LLM이 생성한 설명은 사실과 다를 수 있으므로 화면에 주의 문구를 함께 표시합니다.
