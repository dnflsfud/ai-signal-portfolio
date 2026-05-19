# Streamlit Cloud 배포 가이드

`streamlit_mobile.py` 모바일 대시보드를 무료 Streamlit Community Cloud에
올려서 휴대폰에서 자유롭게 확인하기 위한 단계별 가이드.

## 사전 준비물

- GitHub 계정 (없으면 https://github.com/signup)
- Streamlit Cloud 계정 (https://share.streamlit.io — GitHub 로그인 가능)
- 로컬에서 `python scripts/build_dashboard_data.py`가 정상 실행되어
  `outputs/baseline_v4/dashboard_data.pkl` (~3 MB)이 생성되어 있어야 함

## 배포 페이로드 = 5개 파일

업로드 대상은 `cc2_harness` 디렉터리 루트의 **5개 파일만**:

| 파일 | 크기 | 역할 |
|---|---:|---|
| `streamlit_mobile.py` | ~12 KB | 대시보드 메인 |
| `requirements_dashboard.txt` | ~0.3 KB | 런타임 의존성 (5개 패키지) |
| `outputs/baseline_v4/dashboard_data.pkl` | ~2.6 MB | 미리 계산된 데이터 |
| `.streamlit/config.toml` | ~0.3 KB | 다크 테마 설정 |
| `.gitignore` | ~1 KB | 큰 파일 제외 규칙 |

**`outputs/baseline_v4/backtest_result.pkl` (65 MB)나 `data/` 폴더는 절대 올리지 마세요.**
그건 로컬용. dashboard_data.pkl만 있으면 대시보드는 완전히 돌아갑니다.

## 1단계 — 데이터 빌드

새 백테스트가 끝났거나 baseline이 바뀌었으면 매번:

```bash
cd c2/ai_signal_cc2_harness
python scripts/build_dashboard_data.py
# → outputs/baseline_v4/dashboard_data.pkl  생성 (~3 MB)
```

## 2단계 — GitHub repo 만들기 (private 권장)

대시보드는 **포지션과 시그널을 노출**하므로 private repo 권장.
Streamlit Cloud free tier는 private repo도 무료로 연결 가능.

```bash
# 빈 repo 한 개 만들고 (예: cc2-dashboard)
git init
git remote add origin https://github.com/<your-username>/cc2-dashboard.git

# .gitignore가 이미 보호하고 있으니 src/, outputs/*.pkl, data/ 같은 건
# 자동으로 제외됨. dashboard_data.pkl 만 예외적으로 commit 됨
git add streamlit_mobile.py requirements_dashboard.txt .gitignore .streamlit/
git add -f outputs/baseline_v4/dashboard_data.pkl   # -f 로 ignore 우회

git commit -m "init: cc2 portfolio dashboard"
git branch -M main
git push -u origin main
```

> **확인**: GitHub 웹페이지에서 dashboard_data.pkl 파일이 보여야 함.
> 안 보이면 `git status`로 staged 됐는지 확인 후 `git add -f` 다시.

## 3단계 — Streamlit Cloud 연결

1. https://share.streamlit.io 접속 → "New app"
2. 다음 입력:
   - **Repository**: `<your-username>/cc2-dashboard`
   - **Branch**: `main`
   - **Main file path**: `streamlit_mobile.py`
3. **Advanced settings → Python version**: 3.11 권장
4. **Advanced settings → Requirements**: `requirements_dashboard.txt` (자동 감지 안 되면 수동 지정)
5. "Deploy!" 클릭 → 1-2분 후 빌드 완료, 공개 URL 발급
   (예: `https://<your-username>-cc2-dashboard.streamlit.app`)

## 4단계 — 비밀번호 보호 (강력 권장)

대시보드는 디폴트로 누구나 접속 가능. 비밀번호 보호하려면:

1. Streamlit Cloud 앱 페이지 → 우상단 "⋮" → **Settings** → **Secrets**
2. 다음 입력:
   ```toml
   password = "your-strong-password-here"
   ```
3. Save → 자동 재배포 → 이제 첫 화면에 비밀번호 입력 칸 등장

`streamlit_mobile.py`의 `check_password()` 함수가 `st.secrets["password"]`을
감지하면 자동으로 게이트 활성화. 비밀번호를 안 두면 free access.

## 5단계 — 휴대폰 등록

1. 휴대폰에서 발급된 URL 접속 (예: `https://csos-cc2-dashboard.streamlit.app`)
2. 비밀번호 입력 → 로그인
3. (iPhone) Safari에서 공유 → "홈 화면에 추가" 누르면 앱처럼 사용
4. (Android) Chrome → 점 3개 → "홈 화면에 추가"

## 데이터 갱신 워크플로우

새 백테스트가 끝났을 때:

```bash
# 1) 새 결과로 dashboard_data.pkl 빌드
python scripts/build_dashboard_data.py

# 2) git에 commit & push
git add -f outputs/baseline_v4/dashboard_data.pkl
git commit -m "update: dashboard data $(date +%Y-%m-%d)"
git push

# 3) Streamlit Cloud는 push를 자동 감지해서 30초 안에 재배포
```

## 트러블슈팅

### "Manager App health check failed" / 빌드 실패
`requirements_dashboard.txt`에 빠진 패키지가 있는지 확인. Streamlit Cloud
로그에 "ModuleNotFoundError: ..." 가 보이면 그 패키지 추가.

### `dashboard_data.pkl: file not found`
GitHub에 commit이 안 됐거나 경로가 틀림. Repo의
`outputs/baseline_v4/dashboard_data.pkl` URL을 직접 열어 파일이 있는지 확인.

### `pickle deserialization` 에러 (버전 불일치)
빌드 머신과 Cloud의 pandas/numpy 버전이 너무 다르면 발생.
`requirements_dashboard.txt`에서 `pandas>=2.0` 같은 느슨한 핀을 더 좁게
(예: `pandas==2.2.3`) 잡아주거나, 빌드 머신에서 동일 버전으로 재빌드.

### 메모리 한도 초과 (Streamlit Cloud free = 1 GB)
`dashboard_data.pkl`이 50 MB 이상이면 IC 계산 같은 로직을 더 잘게 쪼개야 함.
현재 ~3 MB이므로 무관.

### 비밀번호 잊어버림
Streamlit Cloud Settings → Secrets에서 직접 수정 가능. 별도 인증 절차 없음.

## 보안 메모

- **이 대시보드는 시그널·포지션·점수를 직접 노출함**. 트레이딩 의사결정에
  영향을 주는 정보이므로 password 보호 + private repo는 거의 필수.
- 비밀번호 외에 IP allow-list 같은 추가 보호가 필요하면 Streamlit Cloud Pro
  ($20/mo)로 업그레이드하거나, 자체 호스팅(Render/Railway/fly.io 등)으로 이전.
- GitHub Personal Access Token 등 민감 정보는 절대 코드/pkl에 넣지 말 것.

## 대안: 빠른 노출 (배포 없이)

Streamlit Cloud 배포가 부담스러우면 ngrok / Tailscale Funnel 같은 터널로
로컬 서버를 외부에 임시 공개 가능:

```bash
# 1) 로컬에서 서버 가동 (이미 돌고 있을 가능성)
streamlit run streamlit_mobile.py --server.address 0.0.0.0 --server.port 8501

# 2) ngrok 별도 터미널에서
ngrok http 8501
# 발급된 https://xxxx-xxx.ngrok-free.app 를 휴대폰에서 열면 됨
```

ngrok free 플랜은 URL이 매번 바뀌고 동시 접속 40명 한도. PC를 끄면 끊김.
24/7 접속이 필요하면 Streamlit Cloud 추천.
