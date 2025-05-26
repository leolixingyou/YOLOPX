import os
from pathlib import Path

def count_files_in_directory(directory_path):
    """
    统计目录中的文件数量
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        int: 文件数量，如果目录不存在返回0
    """
    if not os.path.exists(directory_path):
        return 0
    
    try:
        # 只统计文件，不包括子目录
        files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
        return len(files)
    except Exception as e:
        print(f"警告: 无法读取目录 {directory_path}: {e}")
        return 0

def compare_specific_folders(folder_list):
    """
    对比指定的文件夹列表，检查所有train/文件夹之间以及所有val/文件夹之间是否一致
    
    Args:
        folder_list (list): 文件夹路径列表
    """
    results = {}
    train_counts = []
    val_counts = []
    
    # 收集所有文件夹的统计数据
    for folder_path in folder_list:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在")
            continue
            
        folder_name = os.path.basename(folder_path.rstrip('/'))
        train_path = os.path.join(folder_path, 'train')
        val_path = os.path.join(folder_path, 'val')
        
        train_count = count_files_in_directory(train_path)
        val_count = count_files_in_directory(val_path)
        
        results[folder_name] = {
            'train': train_count,
            'val': val_count
        }
        
        train_counts.append(train_count)
        val_counts.append(val_count)
    
    # 检查一致性
    train_consistent = len(set(train_counts)) <= 1  # 所有train数量相同
    val_consistent = len(set(val_counts)) <= 1      # 所有val数量相同
    
    print_enhanced_comparison_table(results, train_consistent, val_consistent)

def print_enhanced_comparison_table(results, train_consistent, val_consistent):
    """
    以增强表格形式打印对比结果，显示整体一致性
    
    Args:
        results (dict): 统计结果字典
        train_consistent (bool): 所有train文件夹数量是否一致
        val_consistent (bool): 所有val文件夹数量是否一致
    """
    if not results:
        print("没有找到任何有效的文件夹")
        return
    
    # 获取所有文件夹名称并排序
    folder_names = sorted(results.keys())
    
    # 计算列宽度
    max_folder_name_len = max(len(name) for name in folder_names) if folder_names else 10
    col_width = max(max_folder_name_len, 8) + 2
    
    # 打印表头
    print("\n" + "="*100)
    print("文件数量对比表")
    print("="*100)
    
    # 打印列标题
    header = "类型".ljust(8)
    for folder_name in folder_names:
        header += folder_name.ljust(col_width)
    header += "一致性状态".ljust(12)
    print(header)
    print("-" * len(header))
    
    # 打印train行
    train_row = "train".ljust(8)
    for folder_name in folder_names:
        train_count = results[folder_name]['train']
        train_row += str(train_count).ljust(col_width)
    
    # 添加train一致性状态
    train_status = "✓ 一致" if train_consistent else "✗ 不一致"
    train_row += train_status.ljust(12)
    print(train_row)
    
    # 打印val行
    val_row = "val".ljust(8)
    for folder_name in folder_names:
        val_count = results[folder_name]['val']
        val_row += str(val_count).ljust(col_width)
    
    # 添加val一致性状态
    val_status = "✓ 一致" if val_consistent else "✗ 不一致"
    val_row += val_status.ljust(12)
    print(val_row)
    
    print("-" * len(header))
    print("="*100)
    
    # 详细统计摘要
    total_folders = len(results)
    train_counts = [results[folder]['train'] for folder in folder_names]
    val_counts = [results[folder]['val'] for folder in folder_names]
    
    print(f"\n📊 详细统计摘要:")
    print(f"总文件夹数: {total_folders}")
    print()
    
    # Train文件夹分析
    print(f"🔸 Train文件夹分析:")
    print(f"   所有train文件夹数量是否一致: {'✅ 是' if train_consistent else '❌ 否'}")
    if train_consistent:
        if train_counts:
            print(f"   统一文件数量: {train_counts[0]}")
    else:
        print(f"   文件数量范围: {min(train_counts)} ~ {max(train_counts)}")
        print(f"   具体分布:")
        for folder_name in folder_names:
            print(f"     {folder_name}: {results[folder_name]['train']} 个文件")
    
    print()
    
    # Val文件夹分析
    print(f"🔸 Val文件夹分析:")
    print(f"   所有val文件夹数量是否一致: {'✅ 是' if val_consistent else '❌ 否'}")
    if val_consistent:
        if val_counts:
            print(f"   统一文件数量: {val_counts[0]}")
    else:
        print(f"   文件数量范围: {min(val_counts)} ~ {max(val_counts)}")
        print(f"   具体分布:")
        for folder_name in folder_names:
            print(f"     {folder_name}: {results[folder_name]['val']} 个文件")
    
    print()
    
    # 总体结论
    if train_consistent and val_consistent:
        print("🎉 总体结论: 所有文件夹的train/和val/数量都分别保持一致！")
    elif train_consistent:
        print("⚠️  总体结论: train文件夹数量一致，但val文件夹数量不一致")
    elif val_consistent:
        print("⚠️  总体结论: val文件夹数量一致，但train文件夹数量不一致")
    else:
        print("❌ 总体结论: train和val文件夹数量都不一致，需要检查数据")

def scan_folders_and_compare(root_directory):
    """
    扫描根目录下的所有文件夹，对比train/和val/的文件数量
    
    Args:
        root_directory (str): 根目录路径
        
    Returns:
        dict: 包含统计结果的字典
    """
    results = {}
    
    # 获取根目录下的所有子目录
    try:
        subdirs = [d for d in os.listdir(root_directory) 
                  if os.path.isdir(os.path.join(root_directory, d))]
    except Exception as e:
        print(f"错误: 无法读取根目录 {root_directory}: {e}")
        return results
    
    # 遍历每个子目录
    for subdir in subdirs:
        subdir_path = os.path.join(root_directory, subdir)
        train_path = os.path.join(subdir_path, 'train')
        val_path = os.path.join(subdir_path, 'val')
        
        # 统计train和val文件夹中的文件数量
        train_count = count_files_in_directory(train_path)
        val_count = count_files_in_directory(val_path)
        
        # 只记录包含train或val文件夹的目录
        if train_count > 0 or val_count > 0 or os.path.exists(train_path) or os.path.exists(val_path):
            results[subdir] = {
                'train': train_count,
                'val': val_count
            }
    
    return results

if __name__ == "__main__":
    
    specific_folders = [
        "/workspace/bdd100k/images/",
        "/workspace/bdd100k/labels/det_20/",
        "/workspace/bdd100k/labels/drivable/masks/",
        "/workspace/bdd100k/labels/lane/masks/",
    ]

    
    print("正在分析文件夹一致性...")
    compare_specific_folders(specific_folders)