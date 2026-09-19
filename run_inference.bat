@echo off
cd /d "%~dp0"
call "C:\Users\Aayush\venv\Scripts\activate.bat"
python inference_ensemble_tta.py %*
