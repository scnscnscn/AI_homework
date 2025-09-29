import matplotlib.pyplot as plt
import numpy as np 
import json
import pandas as pd
import re
import os


# 设置中文显示（解决乱码和字体警告）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]  # 系统常见中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 确保保存图片的目录存在
save_dir = 'work/result'
try:
    os.makedirs(save_dir, exist_ok=True)
except OSError as e:
    print(f"创建目录'{save_dir}'失败：{e}，程序退出")
    exit()


# 读取嘉宾信息（仅读取一次）
try:
    with open('work/stars_info.json', 'r', encoding='UTF-8') as file:
        stars_data = json.load(file)
except FileNotFoundError:
    print("未找到'work/stars_info.json'文件，程序退出")
    exit()
except json.JSONDecodeError:
    print("'stars_info.json'文件格式错误，程序退出")
    exit()


# -------------------------- 1. 年龄分布柱状图 --------------------------
birth_years = []
for star in stars_data:
    if 'birth_year' in star and star['birth_year']:
        # 提取4位数字（兼容带字符的情况，如"1985年"）
        year_match = re.search(r'\b\d{4}\b', str(star['birth_year']))
        if year_match:
            try:
                birth_year = int(year_match.group())
                birth_years.append(birth_year)
            except:
                continue

if not birth_years:
    print("未找到有效出生年份数据，无法绘制年龄分布图")
else:
    birth_year_counts = pd.Series(birth_years).value_counts().sort_index()

    plt.figure(figsize=(20, 8))
    plt.bar(
        x=birth_year_counts.index,
        height=birth_year_counts.to_numpy(),
        color='#9999ff',
        edgecolor='white'
    )

    plt.xticks(
        birth_year_counts.index, 
        rotation=60,  
        fontsize=10, 
        ha='right',  
        rotation_mode='anchor'  
    )
    plt.yticks(fontsize=12)
    plt.title('《乘风破浪的姐姐》参赛嘉宾出生年份分布', fontsize=16, pad=20)
    plt.xlabel('出生年份', fontsize=14, labelpad=15)  
    plt.ylabel('人数', fontsize=14, labelpad=15)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/bar_birth_year.jpg', dpi=300)
    plt.show()


# -------------------------- 2. 体重分布饼状图 --------------------------
weights = []
for star in stars_data:
    if 'weight' in star and star['weight']:
        weight_match = re.search(r'(\d+\.?\d*)', str(star['weight']))  # 提取数字（支持小数）
        if weight_match:
            try:
                weight = float(weight_match.group(1))
                weights.append(weight)
            except:
                continue

if not weights:
    print("未找到有效体重数据，无法绘制体重分布图")
else:
    bins = [0, 45, 50, 55, float('inf')]
    labels = ['≤45kg', '45~50kg', '50~55kg', '>55kg']
    weight_groups = pd.cut(pd.Series(weights), bins=bins, labels=labels, right=False)
    weight_counts = weight_groups.value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=(10, 10))  
    explode = (0.1, 0.1, 0, 0)

    plt.pie(
        weight_counts,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        pctdistance=0.75, 
        labeldistance=1.1, 
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12},  
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}  
    )
    
    plt.axis('equal')
    plt.title('《乘风破浪的姐姐》参赛嘉宾体重分布', fontsize=16, pad=20) 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{save_dir}/pie_weight.jpg', dpi=300)
    plt.show()


# -------------------------- 3. 身高分布饼状图 --------------------------
heights = []
for star in stars_data:
    if 'height' in star and star['height']:
        height_match = re.search(r'(\d+\.?\d*)', str(star['height']))
        if height_match:
            try:
                height = float(height_match.group(1))
                heights.append(height)
            except:
                continue

if not heights:
    print("未找到有效身高数据，无法绘制身高分布图")
else:
    bins = [0, 165, 170, float('inf')]
    labels = ['≤165cm', '165~170cm', '>170cm']
    height_groups = pd.cut(pd.Series(heights), bins=bins, labels=labels, right=False)
    height_counts = height_groups.value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=(10, 10))  # 增大饼图尺寸
    explode = (0.1, 0.1, 0)
    
    plt.pie(
        height_counts,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        pctdistance=0.75, 
        labeldistance=1.1, 
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12},  
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'} 
    )
    
    plt.axis('equal')
    plt.title('《乘风破浪的姐姐》参赛嘉宾身高分布', fontsize=16, pad=20)  
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{save_dir}/pie_height.jpg', dpi=300)
    plt.show()