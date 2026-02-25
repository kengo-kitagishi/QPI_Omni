@echo off
title QPI Chat Logger
echo =====================================
echo  QPI Chat Logger - Notion自動保存
echo =====================================
echo.
echo Cursorの会話が終了するたびにNotionに自動保存します。
echo このウィンドウを閉じると停止します。
echo.
cd /d "%~dp0"
python scripts/chat_logger.py
pause
