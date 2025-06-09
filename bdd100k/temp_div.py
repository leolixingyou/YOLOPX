import json
import os
from pathlib import Path

def check_missing_images(json_file_path, images_folder):
    """
    检查JSON文件中的图片名称与实际图片文件夹中的图片是否匹配
    
    Args:
        json_file_path (str): JSON文件路径
        images_folder (str): 图片文件夹路径
        
    Returns:
        tuple: (existing_items, missing_images, json_images, actual_images)
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"错误: {json_file_path} 不包含列表格式的数据")
            return [], [], set(), set()
        
        # 获取JSON中的所有图片名称（去掉扩展名）
        json_images = set()
        json_name_to_item = {}  # 用于映射去掉扩展名的文件名到原始条目
        for item in data:
            if 'name' in item:
                # 去掉扩展名进行比较
                name_without_ext = os.path.splitext(item['name'])[0]
                json_images.add(name_without_ext)
                json_name_to_item[name_without_ext] = item
        
        # 获取实际存在的图片文件（去掉扩展名）
        if os.path.exists(images_folder):
            actual_images = set()
            actual_name_to_file = {}  # 用于映射去掉扩展名的文件名到原始文件名
            for file in os.listdir(images_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 去掉扩展名进行比较
                    name_without_ext = os.path.splitext(file)[0]
                    actual_images.add(name_without_ext)
                    actual_name_to_file[name_without_ext] = file
        else:
            print(f"警告: 图片文件夹 {images_folder} 不存在")
            actual_images = set()
            actual_name_to_file = {}
        
        # 找出缺失的图片（基于去掉扩展名的文件名）
        missing_images = json_images - actual_images
        
        # 筛选出存在的条目
        existing_items = []
        for name_without_ext in json_images:
            if name_without_ext in actual_images:
                existing_items.append(json_name_to_item[name_without_ext])
        
        return existing_items, missing_images, json_images, actual_images
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return [], [], set(), set()
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return [], [], set(), set()
    except Exception as e:
        print(f"错误: {e}")
        return [], [], set(), set()

def convert_format(input_json):
    """
    将原始JSON格式转换为嵌套的frames格式
    
    Args:
        input_json (dict): 原始JSON数据
        
    Returns:
        dict: 转换后的JSON数据
    """
    output = {
        'name': input_json['name'].split('.')[0],  # 去掉文件扩展名
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
        # 如果转换失败，至少保证基本结构存在
        pass
    
    return output

def process_existing_items(existing_items, output_folder):
    """
    处理存在的图片条目，将每个条目转换并保存为单独的JSON文件
    
    Args:
        existing_items (list): 存在对应图片的JSON条目列表
        output_folder (str): 输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    processed_count = 0
    
    # 遍历每个存在的条目
    for item in existing_items:
        try:
            # 转换数据格式
            converted_item = convert_format(item)
            
            # 使用转换后的name（已去掉扩展名）
            json_filename = f"{converted_item['name']}.json"
            
            # 完整的输出路径
            output_path = os.path.join(output_folder, json_filename)
            
            # 将转换后的数据写入新的JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_item, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
        except Exception as e:
            print(f"警告: 处理条目 {item.get('name', 'unknown')} 时出错: {e}")
    
    print(f"成功处理了 {processed_count} 个文件")

def print_statistics(dataset_name, json_images, actual_images, missing_images, existing_items):
    """
    打印统计信息
    
    Args:
        dataset_name (str): 数据集名称
        json_images (set): JSON中的图片名称集合（去掉扩展名）
        actual_images (set): 实际存在的图片名称集合（去掉扩展名）
        missing_images (set): 缺失的图片名称集合（去掉扩展名）
        existing_items (list): 存在的条目列表
    """
    print(f"\n=== {dataset_name.upper()} 数据集统计 ===")
    print(f"JSON文件中的图片数量: {len(json_images)}")
    print(f"实际存在的图片数量: {len(actual_images)}")
    print(f"缺失的图片数量: {len(missing_images)}")
    print(f"可以处理的条目数量: {len(existing_items)}")
    print(f"匹配率: {len(existing_items)}/{len(json_images)} ({len(existing_items)/len(json_images)*100:.1f}%)" if json_images else "匹配率: 0%")
    
    if missing_images:
        print(f"\n缺失的图片文件 ({len(missing_images)} 个，按文件名去掉扩展名匹配):")
        # 只显示前10个缺失的文件，避免输出过长
        missing_list = sorted(list(missing_images))
        for i, img in enumerate(missing_list[:10]):
            print(f"  {i+1}. {img}")
        if len(missing_images) > 10:
            print(f"  ... 还有 {len(missing_images) - 10} 个缺失文件")
    else:
        print("✅ 所有图片文件都存在（按文件名匹配）！")

def main():
    """
    主函数，处理train.json和val.json文件 
    """
    root_dir = '/workspace/bdd100k/'
    labels_dir = f'{root_dir}labels/det_20/'
    images_dir = f'{root_dir}labels/lane/masks/'

    # 定义输入文件、图片文件夹和输出文件夹的映射
    file_mappings = {
        f'{labels_dir}det_train.json': {
            'images_folder': f'{images_dir}train/',
            'output_folder': f'{labels_dir}train/',
            'dataset_name': 'train'
        },
        f'{labels_dir}det_val.json': {
            'images_folder': f'{images_dir}val/',
            'output_folder': f'{labels_dir}val/',
            'dataset_name': 'val'
        }
    }
    
    total_missing = 0
    total_processed = 0
    
    for json_file, config in file_mappings.items():
        print(f"\n{'='*50}")
        print(f"正在处理 {json_file}...")
        
        # 检查输入文件是否存在
        if not os.path.exists(json_file):
            print(f"警告: 文件 {json_file} 不存在，跳过处理")
            continue
        
        # 检查图片和JSON的匹配情况
        existing_items, missing_images, json_images, actual_images = check_missing_images(
            json_file, config['images_folder']
        )
        
        # 打印统计信息
        print_statistics(
            config['dataset_name'], 
            json_images, 
            actual_images, 
            missing_images, 
            existing_items
        )
        
        total_missing += len(missing_images)
        
        # 如果有可以处理的条目，询问是否继续处理
        if existing_items:
            print(f"\n是否继续处理 {config['dataset_name']} 数据集的 {len(existing_items)} 个有效条目？")
            user_input = input("输入 'y' 或 'yes' 继续，其他任意键跳过: ").lower().strip()
            
            if user_input in ['y', 'yes']:
                print(f"开始处理 {config['dataset_name']} 数据集...")
                process_existing_items(existing_items, config['output_folder'])
                total_processed += len(existing_items)
                print(f"完成处理 {config['dataset_name']} 数据集")
            else:
                print(f"跳过处理 {config['dataset_name']} 数据集")
        else:
            print(f"没有可处理的条目，跳过 {config['dataset_name']} 数据集")
    
    print(f"\n{'='*50}")
    print("处理总结:")
    print(f"总共缺失图片数量: {total_missing}")
    print(f"总共处理条目数量: {total_processed}")
    print("所有文件处理完成！")


if __name__ == "__main__":
    main()