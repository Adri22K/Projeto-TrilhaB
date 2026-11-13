@echo off
title CADENC.IA - Dashboard Streamlit
cd /d "%~dp0"
echo Iniciando o Dashboard CADENC.IA...
.venv\Scripts\python.exe -m streamlit run app.py
pause

