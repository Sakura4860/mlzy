"""
环境验证脚本
验证所有必要的包是否正确安装
"""

import sys
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否可以导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} 未安装或无法导入")
        print(f"   错误: {e}")
        return False

def main():
    print("="*60)
    print("环境验证检查")
    print("="*60)
    
    # Python版本检查
    print(f"\nPython版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 核心包检查
    print("\n核心依赖包:")
    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('openpyxl', 'openpyxl'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('torch', 'torch'),
        ('jupyter', 'jupyter'),
        ('joblib', 'joblib'),
        ('tqdm', 'tqdm'),
    ]
    
    success_count = 0
    for pkg_name, import_name in packages:
        if check_package(pkg_name, import_name):
            success_count += 1
    
    # 项目结构检查
    print("\n项目结构检查:")
    required_dirs = [
        'src',
        'src/models',
        'data',
        'notebooks',
        'results',
        'results/figures',
        'results/models',
        'results/metrics'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path:30s} 存在")
        else:
            print(f"❌ {dir_path:30s} 不存在")
    
    # 数据文件检查
    print("\n数据文件检查:")
    data_files = [
        'data/WEATHER_DATA_ZURICH_2020_2019.xlsx',
        'data/Hospitals_1991_2000_Full_retrofit.xlsx',
        'data/Restaurants_1991_2000_Full_retrofit.xlsx',
        'data/Schools_2010_2015_Full_retrofit.xlsx',
        'data/Shops_1991_2000_Full_retrofit.xlsx'
    ]
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {path.name:45s} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {path.name:45s} 不存在")
    
    # PyTorch特殊检查
    print("\nPyTorch信息:")
    try:
        import torch
        print(f"  版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
        else:
            print(f"  运行模式: CPU (适合本项目)")
    except Exception as e:
        print(f"  ❌ PyTorch检查失败: {e}")
    
    # 总结
    print("\n" + "="*60)
    print(f"检查完成: {success_count}/{len(packages)} 包可用")
    
    if success_count == len(packages):
        print("✅ 环境设置完美！可以开始运行项目了。")
        print("\n下一步:")
        print("  1. 运行完整实验: python main.py")
        print("  2. 交互式探索: jupyter notebook")
    else:
        print("⚠️  有些包未正确安装，请检查上述错误信息。")
        print("   尝试重新安装: pip install -r requirements.txt")
    
    print("="*60)

if __name__ == '__main__':
    main()
