@echo off
rem Starts the dashboard using the venv of the speed_signs project (this folder sits inside speed_signs)
call ..\venv\Scripts\activate.bat
python run.py
pause
