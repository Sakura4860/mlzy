# PowerShell虚拟环境激活脚本
# 如果PowerShell执行策略限制，请使用 activate_env.bat

Write-Host "正在激活虚拟环境..." -ForegroundColor Green

# 激活虚拟环境
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "虚拟环境已激活！" -ForegroundColor Green
Write-Host "Python版本："
& python --version

Write-Host ""
Write-Host "已安装的主要包："
& pip list | Select-String "numpy|pandas|scikit-learn|torch|matplotlib"

Write-Host ""
Write-Host "提示：" -ForegroundColor Yellow
Write-Host "- 运行项目: python main.py"
Write-Host "- 启动Jupyter: jupyter notebook"
Write-Host "- 退出虚拟环境: deactivate"
Write-Host ""
