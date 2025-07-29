import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

def get_files_without_extension(folder_path, extensions=None):
    """
    获取文件夹中的文件名（去掉扩展名）
    
    Args:
        folder_path (str): 文件夹路径
        extensions (list): 允许的扩展名列表，如果为None则获取所有文件
        
    Returns:
        dict: {filename_without_ext: full_filename}
    """
    files_dict = {}
    if not os.path.exists(folder_path):
        print(f"警告: 文件夹 {folder_path} 不存在")
        return files_dict
    
    try:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if extensions is None or any(file.lower().endswith(ext) for ext in extensions):
                    name_without_ext = os.path.splitext(file)[0]
                    files_dict[name_without_ext] = file
    except Exception as e:
        print(f"错误: 读取文件夹 {folder_path} 时出错: {e}")
    
    return files_dict

def get_json_files_info(json_file_path):
    """
    从JSON文件中获取图片名称信息
    
    Args:
        json_file_path (str): JSON文件路径
        
    Returns:
        dict: {filename_without_ext: json_item}
    """
    files_dict = {}
    if not os.path.exists(json_file_path):
        print(f"警告: JSON文件 {json_file_path} 不存在")
        return files_dict
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if 'name' in item:
                    name_without_ext = os.path.splitext(item['name'])[0]
                    files_dict[name_without_ext] = item
    except Exception as e:
        print(f"错误: 读取JSON文件 {json_file_path} 时出错: {e}")
    
    return files_dict

def find_common_files(base_dir):
    """
    找到在所有数据源中都存在的文件
    
    Args:
        base_dir (str): 基础目录路径
        
    Returns:
        dict: 包含train和val的匹配文件信息
    """
    result = {}
    
    for split in ['train', 'val']:
        print(f"\n正在分析 {split} 数据集...")
        
        # 获取各个数据源中的文件
        sources = {
            'images': get_files_without_extension(
                f'{base_dir}/images/{split}/', 
                ['.jpg', '.jpeg', '.png']
            ),
            'det_json': get_json_files_info(f'{base_dir}/labels/det_20/det_{split}.json'),
            'drivable': get_files_without_extension(
                f'{base_dir}/labels/drivable/masks/{split}/', 
                ['.png', '.jpg', '.jpeg']
            ),
            'lane': get_files_without_extension(
                f'{base_dir}/labels/lane/masks/{split}/', 
                ['.png', '.jpg', '.jpeg']
            )
        }
        
        # 打印每个数据源的统计信息
        for source_name, source_files in sources.items():
            print(f"  {source_name}: {len(source_files)} 个文件")
        
        # 找到所有数据源的交集
        all_file_sets = [set(source_files.keys()) for source_files in sources.values()]
        common_files = set.intersection(*all_file_sets) if all_file_sets else set()
        
        print(f"  匹配的文件数量: {len(common_files)}")
        
        # 构建匹配文件的完整信息
        matched_files = {}
        for filename in common_files:
            matched_files[filename] = {
                'images': sources['images'][filename],
                'det_json': sources['det_json'][filename],
                'drivable': sources['drivable'][filename],
                'lane': sources['lane'][filename]
            }
        
        result[split] = {
            'matched_files': matched_files,
            'sources_stats': {name: len(files) for name, files in sources.items()},
            'common_count': len(common_files)
        }
    
    return result

def convert_format(input_json):
    """
    将原始JSON格式转换为嵌套的frames格式
    """
    output = {
        'name': input_json['name'].split('.')[0],
        'frames': [{
            'timestamp': input_json['timestamp'],
            'objects': []
        }],
        'attributes': input_json['attributes']
    }
    
    try:
        for label in input_json['labels']:
            obj = {
                'category': label['category'],
                'id': int(label['id']),
                'attributes': {
                    'occluded': label['attributes']['occluded'],
                    'truncated': label['attributes']['truncated'],
                    'trafficLightColor': 'green' if label['attributes']['trafficLightColor'] == 'G' else 'none'
                }
            }
            if 'box2d' in label:
                obj['box2d'] = label['box2d']
            output['frames'][0]['objects'].append(obj)
    except Exception as e:
        print(f"警告: 转换labels时出错: {e}")
    
    return output

def copy_matched_files(base_dir, matched_data, output_dir):
    """
    复制匹配的文件到输出目录
    
    Args:
        base_dir (str): 源目录
        matched_data (dict): 匹配的文件数据
        output_dir (str): 输出目录
    """
    print(f"\n开始复制文件到 {output_dir}...")
    
    # 创建输出目录结构
    dirs_to_create = [
        f'{output_dir}/images/train',
        f'{output_dir}/images/val',
        f'{output_dir}/labels/det_20/train',
        f'{output_dir}/labels/det_20/val',
        f'{output_dir}/labels/drivable/masks/train',
        f'{output_dir}/labels/drivable/masks/val',
        f'{output_dir}/labels/lane/masks/train',
        f'{output_dir}/labels/lane/masks/val'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    total_copied = 0
    
    for split in ['train', 'val']:
        matched_files = matched_data[split]['matched_files']
        print(f"\n复制 {split} 数据集 ({len(matched_files)} 个文件)...")
        
        for filename, file_info in matched_files.items():
            try:
                # 复制图片文件
                src_image = f'{base_dir}/images/{split}/{file_info["images"]}'
                dst_image = f'{output_dir}/images/{split}/{file_info["images"]}'
                if os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                
                # 创建并保存转换后的JSON文件
                converted_item = convert_format(file_info['det_json'])
                json_path = f'{output_dir}/labels/det_20/{split}/{filename}.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_item, f, ensure_ascii=False, indent=2)
                
                # 复制drivable mask
                src_drivable = f'{base_dir}/labels/drivable/masks/{split}/{file_info["drivable"]}'
                dst_drivable = f'{output_dir}/labels/drivable/masks/{split}/{file_info["drivable"]}'
                if os.path.exists(src_drivable):
                    shutil.copy2(src_drivable, dst_drivable)
                
                # 复制lane mask
                src_lane = f'{base_dir}/labels/lane/masks/{split}/{file_info["lane"]}'
                dst_lane = f'{output_dir}/labels/lane/masks/{split}/{file_info["lane"]}'
                if os.path.exists(src_lane):
                    shutil.copy2(src_lane, dst_lane)
                
                total_copied += 1
                
                if total_copied % 100 == 0:
                    print(f"  已复制 {total_copied} 个文件...")

            except Exception as e:
                print(f"警告: 复制文件 {filename} 时出错: {e}")
    
    print(f"\n复制完成！总共复制了 {total_copied} 个完整的文件集合")

def print_summary(matched_data):
    """
    打印汇总统计信息
    """
    print("\n" + "="*60)
    print("数据匹配汇总")
    print("="*60)
    
    total_common = 0
    for split in ['train', 'val']:
        data = matched_data[split]
        common_count = data['common_count']
        total_common += common_count
        
        print(f"\n{split.upper()} 数据集:")
        print(f"  Images: {data['sources_stats']['images']} 个文件")
        print(f"  Detection JSON: {data['sources_stats']['det_json']} 个条目")
        print(f"  Drivable masks: {data['sources_stats']['drivable']} 个文件")
        print(f"  Lane masks: {data['sources_stats']['lane']} 个文件")
        print(f"  → 匹配文件数: {common_count} 个")
    
    print(f"\n总计匹配文件数: {total_common}")
    
    return total_common

def main():
    """
    主函数
    """
    # 基础目录路径
    base_dir = '/workspace/bdd100k'
    
    print("BDD100K 数据集文件匹配和复制工具")
    print("="*60)
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 基础目录 {base_dir} 不存在")
        return
    
    # 找到匹配的文件
    matched_data = find_common_files(base_dir)
    
    # 打印汇总信息
    total_matched = print_summary(matched_data)
    
    if total_matched == 0:
        print("\n没有找到匹配的文件，程序结束。")
        return
    
    # 询问是否继续复制
    print(f"\n找到 {total_matched} 个完全匹配的文件。")
    user_input = input("是否创建新的数据集？输入 'y' 或 'yes' 继续: ").lower().strip()
    
    if user_input in ['y', 'yes']:
        # 创建输出目录名称
        output_dir = f'/workspace/bdd100k_{total_matched}'
        
        print(f"\n将创建新数据集: {output_dir}")
        
        # 检查输出目录是否已存在
        if os.path.exists(output_dir):
            overwrite = input(f"目录 {output_dir} 已存在，是否覆盖？输入 'y' 继续: ").lower().strip()
            if overwrite != 'y':
                print("操作取消。")
                return
            shutil.rmtree(output_dir)
        
        # 复制文件
        copy_matched_files(base_dir, matched_data, output_dir)
        
        print(f"\n✅ 数据集创建完成！")
        print(f"新数据集路径: {output_dir}")
        print(f"包含 {total_matched} 个完整的文件集合")
        
    else:
        print("操作取消。")

if __name__ == "__main__":
    main()