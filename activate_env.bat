@echo off
REM 激活虚拟环境的批处理脚本

echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 虚拟环境已激活！
echo Python版本：
python --version
echo.
echo 已安装的主要包：
pip list | findstr "numpy pandas scikit-learn torch matplotlib"
echo.
echo 提示：
echo - 运行项目: python main.py
echo - 启动Jupyter: jupyter notebook
echo - 退出虚拟环境: deactivate
echo.
