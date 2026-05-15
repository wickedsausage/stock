@echo off
chcp 65001 >nul
echo ========================================
echo   飞书双向机器人启动中...
echo ========================================
echo.

cd /d "%~dp0feishu-server"

:: 检查 .env 文件
if not exist .env (
    echo [错误] 未找到 .env 文件！
    echo 请复制 .env.example 为 .env 并填入你的密钥
    pause
    exit /b 1
)

:: 检查 ngrok
where ngrok >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] ngrok 未安装，正在下载...
    powershell -Command "Invoke-WebRequest -Uri 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip' -OutFile '%TEMP%\ngrok.zip'" && powershell -Command "Expand-Archive -Path '%TEMP%\ngrok.zip' -DestinationPath '%TEMP%\ngrok'" && copy /y "%TEMP%\ngrok\ngrok.exe" "%~dp0ngrok.exe" >nul
    if %errorlevel% neq 0 (
        echo [错误] ngrok 下载失败，请手动下载放到项目根目录
        pause
        exit /b 1
    )
)

echo [1/2] 启动 ngrok 隧道 (端口 5000)...
start "ngrok" "%~dp0ngrok.exe" http 5000

:: 等待 ngrok 启动
timeout /t 3 /nobreak >nul

echo [2/2] 启动 Flask 服务器...
python server.py

pause
