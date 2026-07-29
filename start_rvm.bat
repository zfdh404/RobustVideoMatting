@echo off
chcp 65001 > nul

title RobustVideoMatting Launcher


echo ========================================
echo   启动 RobustVideoMatting
echo ========================================
echo.


echo [1/5] 激活 conda 环境...

call D:\ProgramFiles\miniconda3\Scripts\activate.bat rvm


echo [2/5] 切换到项目目录...

cd /d D:\develop\RobustVideoMatting


echo [3/5] 清除代理设置...

set NO_PROXY=localhost,127.0.0.1
set no_proxy=localhost,127.0.0.1

set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=



echo [4/5] 启动应用程序...


:: 延迟5秒打开浏览器, app.py中已实现
:: start "" cmd /c "timeout /t 5 >nul && start http://127.0.0.1:7860"


:: 前台运行，显示日志
python app.py


pause