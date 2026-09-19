@echo off
cd /d "%~dp0"
call "C:\Users\Aayush\venv\Scripts\activate.bat"
python train_kfold.py --epochs 20 --batch_size 64 %*
