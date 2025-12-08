# 虚拟环境使用指南

## 虚拟环境已创建成功！ ✅

### 激活虚拟环境

#### Windows PowerShell
```powershell
# 方法1：直接激活（如果未被策略限制）
.\venv\Scripts\Activate.ps1

# 方法2：使用批处理文件
.\activate_env.bat

# 方法3：如果PowerShell策略限制，临时允许脚本执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\activate_env.ps1
```

#### Windows CMD
```cmd
.\venv\Scripts\activate.bat
```

### 使用虚拟环境中的Python

激活后，直接使用：
```bash
python main.py
jupyter notebook
```

或者不激活，直接使用完整路径：
```bash
.\venv\Scripts\python.exe main.py
.\venv\Scripts\jupyter.exe notebook
```

### 已安装的包

核心包：
- numpy (2.3.5)
- pandas (2.3.3)
- scikit-learn (1.7.2)
- matplotlib (3.10.7)
- seaborn (0.13.2)
- torch (2.9.1+cpu)
- jupyter
- ipykernel

其他依赖包已自动安装。

### 验证安装

```python
# 测试导入
python -c "import numpy, pandas, sklearn, torch, matplotlib; print('所有包导入成功！')"
```

### 退出虚拟环境

```bash
deactivate
```

### VSCode集成

1. 打开命令面板（Ctrl+Shift+P）
2. 输入 "Python: Select Interpreter"
3. 选择 `.\venv\Scripts\python.exe`

### Jupyter Notebook内核

虚拟环境会自动作为Jupyter内核可用。如需手动注册：
```bash
.\venv\Scripts\python.exe -m ipykernel install --user --name=mlzy --display-name="ML Energy Prediction"
```

### 常见问题

**Q: PowerShell无法运行脚本？**
A: 使用 `activate_env.bat` 或在CMD中激活。

**Q: 如何更新包？**
A: 激活环境后运行 `pip install --upgrade <package-name>`

**Q: 如何重新创建环境？**
A: 
```bash
# 删除旧环境
Remove-Item -Recurse -Force venv

# 创建新环境
python -m venv venv

# 重新安装依赖
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 下一步

1. 激活虚拟环境
2. 运行 `python main.py` 开始实验
3. 或打开 `jupyter notebook` 进行交互式探索

祝实验顺利！🚀
