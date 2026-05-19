@echo off
REM =====================================================================
REM   AI Signal — one-click update & deploy
REM   더블클릭 실행용 래퍼. 인자 없이 실행하면:
REM     1) 데이터 freshness 점검 (마지막 행 날짜 vs 오늘)
REM     2) production variant (iter15_65tkr_reb21_vtg) 풀 백테스트 (~3-4분)
REM     3) outputs/<variant>/ → outputs/baseline_v4/ 자동 promote
REM     4) dashboard_data.pkl 빌드
REM     5) cc2-dashboard repo 로 git push → Streamlit Cloud 자동 재배포
REM   빠른 증분 모드 (백테스트 pkl 갱신 X):  update_and_deploy.bat --mode incremental
REM =====================================================================

setlocal
cd /d "%~dp0"

REM 사용 중인 venv 자동 활성화 (있으면)
if exist "..\..\..\..\Scripts\activate.bat" (
    call "..\..\..\..\Scripts\activate.bat"
)

python update_and_deploy.py %*
set ERR=%ERRORLEVEL%

if %ERR% NEQ 0 (
    echo.
    echo [ERROR] update_and_deploy.py exited with code %ERR%
    pause
    exit /b %ERR%
)

echo.
echo [OK] Pipeline finished. Streamlit Cloud will redeploy in ~30s.
pause
