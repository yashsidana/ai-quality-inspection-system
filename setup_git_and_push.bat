@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  AI Quality Inspection System - GitHub Push Automation
echo =====================================================================
echo.

cd /d "%~dp0"

echo [1/4] Generating synthetic manufacturing sample images...
python data\generate_samples.py

echo.
echo [2/4] Initializing local Git repository...
if not exist ".git" (
    git init
    echo Git repository initialized.
) else (
    echo Git repository already initialized.
)

echo.
echo [3/4] Staging files and creating commit...
git add .
git commit -m "feat: AI-Based Quality Inspection System with YOLOv8, OpenCV, FastAPI, PostgreSQL & Streamlit"
git branch -M main

echo.
echo [4/4] Setting GitHub remote and pushing to repository...
echo Target Repository: https://github.com/yashsidana/ai-quality-inspection-system.git
echo.

git remote remove origin 2>nul
git remote add origin https://github.com/yashsidana/ai-quality-inspection-system.git

echo Pushing to GitHub main branch...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo =====================================================================
    echo  SUCCESS: Code pushed to GitHub!
    echo  Repo URL: https://github.com/yashsidana/ai-quality-inspection-system
    echo =====================================================================
    echo.
    echo Next Step: Deploy to Streamlit Community Cloud (share.streamlit.io)
    echo Refer to deploy_streamlit_guide.md for 1-click deployment instructions.
) else (
    echo.
    echo =====================================================================
    echo  NOTE: If push was rejected, please ensure:
    echo  1. You created the repository 'ai-quality-inspection-system' on https://github.com/new
    echo  2. If prompted for password, use your GitHub Personal Access Token (PAT).
    echo =====================================================================
)

echo.
pause
