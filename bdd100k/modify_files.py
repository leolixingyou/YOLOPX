import os
import shutil
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

def get_filenames_in_directory(directory_path):
    """
    获取目录中所有文件名的集合（去掉扩展名）
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        set: 文件名集合（不包含扩展名）
    """
    if not os.path.exists(directory_path):
        return set()
    
    try:
        files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
        # 去掉扩展名，只保留基础文件名
        base_names = set(Path(f).stem for f in files)
        return base_names
    except Exception as e:
        print(f"警告: 无法读取目录 {directory_path}: {e}")
        return set()

def get_full_filenames_in_directory(directory_path):
    """
    获取目录中所有完整文件名的字典，以基础文件名为key
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        dict: {基础文件名: 完整文件名}
    """
    if not os.path.exists(directory_path):
        return {}
    
    try:
        files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f))]
        # 创建基础文件名到完整文件名的映射
        name_mapping = {Path(f).stem: f for f in files}
        return name_mapping
    except Exception as e:
        print(f"警告: 无法读取目录 {directory_path}: {e}")
        return {}

def analyze_file_matching(folder_list, split_type):
    """
    分析指定split（train或val）中所有文件夹的文件名匹配情况
    
    Args:
        folder_list (list): 文件夹路径列表
        split_type (str): 'train' 或 'val'
        
    Returns:
        tuple: (intersection_set, folder_file_sets, differences)
    """
    folder_file_sets = {}
    
    # 收集每个文件夹中该split的所有文件名
    for folder_path in folder_list:
        if not os.path.exists(folder_path):
            continue
            
        folder_name = os.path.basename(folder_path.rstrip('/'))
        split_path = os.path.join(folder_path, split_type)
        
        filenames = get_filenames_in_directory(split_path)
        folder_file_sets[folder_name] = filenames
    
    if not folder_file_sets:
        return set(), {}, {}
    
    # 计算交集（所有文件夹都有的文件）
    intersection = set.intersection(*folder_file_sets.values()) if folder_file_sets else set()
    
    # 计算每个文件夹与交集的差异
    differences = {}
    for folder_name, file_set in folder_file_sets.items():
        differences[folder_name] = file_set - intersection
    
    return intersection, folder_file_sets, differences

def copy_matching_files(folder_list, modified_folder, intersection_train, intersection_val, num_set):
    """
    num_set: [num_train, num_val]
    将匹配的文件复制到新的目录结构中
    
    Args:
        folder_list (list): 原始文件夹路径列表
        modified_folder (str): 目标根目录
        intersection_train (set): train文件的交集（基础文件名）
        intersection_val (set): val文件的交集（基础文件名）
    """
    print(f"\n🔄 开始复制文件到 {modified_folder}")
    
    copied_counts = {"train": 0, "val": 0}
    
    for folder_path in folder_list:
        folder_name = os.path.basename(folder_path.rstrip('/'))
        
        # 确定目标路径
        if folder_name == "images":
            target_subpath = "images"
        elif folder_name == "det_20":
            target_subpath = "labels/det_20"
        elif folder_name == "masks":
            if "drivable" in folder_path:
                target_subpath = "labels/drivable/masks"
            elif "lane" in folder_path:
                target_subpath = "labels/lane/masks"
            else:
                target_subpath = f"labels/{folder_name}"
        else:
            target_subpath = f"labels/{folder_name}"
        
        target_folder = os.path.join(modified_folder, target_subpath)
        
        # 创建目标目录
        os.makedirs(os.path.join(target_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, "val"), exist_ok=True)
        
        print(f"   📂 处理文件夹: {folder_name} -> {target_subpath}")
        
        # 复制train文件
        if intersection_train:
            source_train = os.path.join(folder_path, "train")
            target_train = os.path.join(target_folder, "train")
            
            # 获取源文件夹中的文件名映射
            train_file_mapping = get_full_filenames_in_directory(source_train)
            
            train_copied = 0
            for i, base_name in enumerate(intersection_train):
                if base_name in train_file_mapping and i<= num_set[0]:
                    full_filename = train_file_mapping[base_name]
                    source_file = os.path.join(source_train, full_filename)
                    target_file = os.path.join(target_train, full_filename)
                    
                    if os.path.exists(source_file):
                        try:
                            shutil.copy2(source_file, target_file)
                            train_copied += 1
                        except Exception as e:
                            print(f"      ⚠️ 复制失败 {full_filename}: {e}")
            
            print(f"      ✅ Train: {train_copied}/{len(intersection_train)} 个文件")
            if folder_name == "images":  # 只在images文件夹统计总数
                copied_counts["train"] = train_copied
        
        # 复制val文件
        if intersection_val:
            source_val = os.path.join(folder_path, "val")
            target_val = os.path.join(target_folder, "val")
            
            # 获取源文件夹中的文件名映射
            val_file_mapping = get_full_filenames_in_directory(source_val)
            
            val_copied = 0
            for i, base_name in enumerate(intersection_val) :
                if base_name in val_file_mapping and i<= num_set[1]:
                    full_filename = val_file_mapping[base_name]
                    source_file = os.path.join(source_val, full_filename)
                    target_file = os.path.join(target_val, full_filename)
                    
                    if os.path.exists(source_file):
                        try:
                            shutil.copy2(source_file, target_file)
                            val_copied += 1
                        except Exception as e:
                            print(f"      ⚠️ 复制失败 {full_filename}: {e}")
            
            print(f"      ✅ Val: {val_copied}/{len(intersection_val)} 个文件")
            if folder_name == "images":  # 只在images文件夹统计总数
                copied_counts["val"] = val_copied
    
    print(f"\n📋 复制完成统计:")
    print(f"   Train文件: {len(intersection_train)} 个")
    print(f"   Val文件: {len(intersection_val)} 个")
    print(f"   总计: {len(intersection_train) + len(intersection_val)} 个文件")
    print(f"   所有文件夹都已复制对应的匹配文件（保持各自的扩展名）！")

def compare_specific_folders(folder_list, modified_folder, num_set):
    """
    对比指定的文件夹列表，检查所有train/文件夹之间以及所有val/文件夹之间是否一致
    
    Args:
        folder_list (list): 文件夹路径列表
        modified_folder (str): 修改后文件的保存目录
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
    
    # 无论是否一致，都进行文件匹配分析和复制
    print(f"\n🔍 详细文件匹配分析:")
    
    # 分析train文件匹配情况
    print(f"\n📁 Train文件夹匹配分析:")
    intersection_train, train_sets, train_diffs = analyze_file_matching(folder_list, "train")
    print(f"   所有文件夹共同拥有的文件数: {len(intersection_train)}")
    
    for folder_name, diff_set in train_diffs.items():
        if diff_set:
            print(f"   {folder_name} 独有的文件数: {len(diff_set)}")
        else:
            print(f"   {folder_name} 无独有文件")
    
    # 分析val文件匹配情况
    print(f"\n📁 Val文件夹匹配分析:")
    intersection_val, val_sets, val_diffs = analyze_file_matching(folder_list, "val")
    print(f"   所有文件夹共同拥有的文件数: {len(intersection_val)}")
    
    for folder_name, diff_set in val_diffs.items():
        if diff_set:
            print(f"   {folder_name} 独有的文件数: {len(diff_set)}")
        else:
            print(f"   {folder_name} 无独有文件")
    
    # 总是复制匹配的文件
    if intersection_train or intersection_val:
        copy_matching_files(folder_list, modified_folder, intersection_train, intersection_val, num_set)
    else:
        print(f"\n❌ 没有找到任何匹配的文件，无法创建修改后的数据集")

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
    print("\n文件数量对比表")
    print("="*100)
    
    # 打印列标题
    header = "类型".ljust(8)
    for folder_name in folder_names:
        header += folder_name.ljust(col_width)
    header += "一致性状态".ljust(12)
    print(header)
    print("-" * 50)
    
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
    
    print("-" * 50)
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

if __name__ == "__main__":
    num_train, num_val = 1000, 100
    num_set = [num_train, num_val]
    specific_folders = [
        "/workspace/bdd100k/images/",
        "/workspace/bdd100k/labels/det_20/",
        "/workspace/bdd100k/labels/drivable/masks/",
        "/workspace/bdd100k/labels/lane/masks/",
    ]

    modified_folder = "/workspace/modified_bdd100k_1000/"
    
    print("正在分析文件夹一致性...")
    compare_specific_folders(specific_folders, modified_folder, num_set)