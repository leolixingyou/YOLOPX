import json
import os
from pathlib import Path

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

def process_json_file(input_file_path, output_folder):
    """
    处理JSON文件，将每个条目按name字段拆分成单独的JSON文件
    
    Args:
        input_file_path (str): 输入JSON文件路径
        output_folder (str): 输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # 读取JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保data是一个列表
        if not isinstance(data, list):
            print(f"错误: {input_file_path} 不包含列表格式的数据")
            return
        
        # 遍历每个条目
        for item in data:
            if 'name' not in item:
                print(f"警告: 跳过没有'name'字段的条目: {item}")
                continue
            
            # 转换数据格式
            converted_item = convert_format(item)
            
            # 使用转换后的name（已去掉扩展名）
            json_filename = f"{converted_item['name']}.json"
            
            # 完整的输出路径
            output_path = os.path.join(output_folder, json_filename)
            
            # 将转换后的数据写入新的JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_item, f, ensure_ascii=False, indent=2)
            
            print(f"已创建: {output_path}")
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
    except Exception as e:
        print(f"错误: {e}")

def main():
    """
    主函数，处理train.json和val.json文件
    """
    root_dir = '/workspace/bdd100k/labels/det_20/'

    # 定义输入文件和输出文件夹的映射
    file_mappings = {
        f'{root_dir}det_train.json': f'{root_dir}train/',
        f'{root_dir}det_val.json': f'{root_dir}val/'
    }
    
    for input_file, output_folder in file_mappings.items():
        print(f"\n正在处理 {input_file}...")
        
        # 检查输入文件是否存在
        if os.path.exists(input_file):
            process_json_file(input_file, output_folder)
            print(f"完成处理 {input_file}")
        else:
            print(f"警告: 文件 {input_file} 不存在，跳过处理")
    
    print("\n所有文件处理完成！")


if __name__ == "__main__":
    main()