import json
import requests
from bs4 import BeautifulSoup
import os
import re
import time  # 补充导入time模块（解决“未定义time”）
import random  # 补充导入random模块（解决“未定义random”）


# -------------------------- 补充缺失的辅助函数 --------------------------
def clean_baike_url(url):
    """清理百度百科链接，移除fromModule等反爬参数（解决“未定义clean_baike_url”）"""
    if isinstance(url, str) and '?' in url:
        url = url.split('?')[0]  # 只保留基础链接（如https://baike.baidu.com/item/阿朵）
    return url


def get_random_headers():
    """生成随机请求头，模拟真实浏览器（解决“未定义get_random_headers”）"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/129.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/128.0.0.0 Safari/537.36'
    ]
    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': 'https://baike.baidu.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    }
    return headers


# -------------------------- 原有函数修正 --------------------------
def crawl_wiki_data():
    """爬取百度百科《乘风破浪的姐姐》嘉宾信息表格，适配页面结构变化"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    url = 'https://baike.baidu.com/item/乘风破浪的姐姐'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        tables = soup.find_all('table')
        target_keywords = ['参赛嘉宾']

        for table in tables:
            title_tags = table.find_previous(['h3', 'div', 'p'])
            if not title_tags:
                continue
            title_text = title_tags.get_text(strip=True).lower()
            if any(keyword in title_text for keyword in target_keywords):
                if table.find('a', href=re.compile(r'/item/')):
                    print(f"找到目标表格，标题包含关键词：{title_text}")
                    return table

        for table in tables:
            a_tags = table.find_all('a', href=re.compile(r'/item/'))
            if len(a_tags) > 5:
                print("通过表格特征找到疑似嘉宾表格")
                return table

        print("未找到嘉宾信息表格（页面结构可能已变更）")
        return None

    except Exception as e:
        print(f"爬取百科表格失败：{str(e)}")
        return None


def parse_wiki_data(table_html):
    """解析嘉宾表格，提取姓名和百科链接"""
    if table_html is None:
        print("无有效表格数据，无法解析")
        return

    bs = BeautifulSoup(str(table_html), 'lxml')
    all_trs = bs.find_all('tr')
    stars = []

    for tr in all_trs:
        all_tds = tr.find_all('td')
        for td in all_tds:
            star = {}
            a_tag = td.find_next('a')
            if a_tag is not None:
                star["name"] = a_tag.get_text(strip=True, separator='')
                href = a_tag.get('href')
                star['link'] = f'https://baike.baidu.com{href}' if (isinstance(href, str) and href) else ''
                stars.append(star)
                continue
            
            div_tag = td.find_next('div')
            if div_tag is not None:
                div_a_tag = div_tag.find('a')
                if div_a_tag is not None:
                    star["name"] = div_a_tag.get_text(strip=True, separator='')
                    href = div_a_tag.get('href')
                    star['link'] = f'https://baike.baidu.com{href}' if (isinstance(href, str) and href) else ''
                    stars.append(star)

    os.makedirs('work', exist_ok=True)
    with open('work/stars.json', 'w', encoding='UTF-8') as f:
        json.dump(stars, f, ensure_ascii=False, indent=2)
    print(f"已解析{len(stars)}位嘉宾信息，保存至stars.json")


def crawl_everyone_wiki_urls():
    """爬取每位嘉宾的详细信息和图片（适配新结构+修复所有报错）"""
    star_json_path = 'work/stars.json'
    if not os.path.exists(star_json_path):
        print(f"{star_json_path}不存在，无法爬取嘉宾详情")
        return
    pic_root_dir = 'work/star_pics'
    os.makedirs(pic_root_dir, exist_ok=True)

    # 读取并过滤嘉宾列表
    with open(star_json_path, 'r', encoding='UTF-8') as file:
        json_array = json.load(file)
    valid_stars = []
    for star in json_array:
        name = star.get('name', '').strip()
        link = star.get('link', '').strip()
        if name and link:
            clean_link = clean_baike_url(link)
            valid_stars.append({'name': name, 'link': clean_link})
    if not valid_stars:
        print("无有效嘉宾列表数据，终止爬取")
        return

    # 初始化会话
    session = requests.Session()
    star_infos = []
    request_interval = (1.5, 3)

    for idx, star in enumerate(valid_stars, 1):
        name = star['name']
        link = star['link']
        star_info = {'name': name, 'link': link}
        print(f"\n[{idx}/{len(valid_stars)}] 开始爬取嘉宾：{name}")

        try:
            # 随机间隔
            time.sleep(random.uniform(*request_interval))
            # 获取个人页面
            headers = get_random_headers()
            response = session.get(link, headers=headers, timeout=20)
            response.raise_for_status()
            bs = BeautifulSoup(response.text, 'lxml')

            # 提取基本信息（新结构）
            base_info_div = bs.find('div', class_='basicInfo_rZDFN J-basic-info')
            if base_info_div:
                item_wrappers = base_info_div.find_all('div', class_='itemWrapper_u4OET')
                for wrapper in item_wrappers:
                    dt_tag = wrapper.find('dt', class_='basicInfoItem_TsAAR itemName_tPWxP')
                    dd_tag = wrapper.find('dd', class_='basicInfoItem_TsAAR itemValue_Xsiqq')
                    if not (dt_tag and dd_tag):
                        continue

                    # 清理字段名
                    dt_text = dt_tag.get_text(strip=True)
                    dt_text = dt_text.replace('\u00A0', '').replace('：', '')
                    # 提取字段值
                    value_span = dd_tag.find('span', class_='text_zBf3n')
                    dd_text = value_span.get_text(strip=True, separator=' ') if value_span else dd_tag.get_text(strip=True, separator=' ')

                    # 赋值目标字段
                    if dt_text == '民族':
                        star_info['nation'] = dd_text
                    elif dt_text == '星座':
                        star_info['constellation'] = dd_text
                    elif dt_text == '血型':
                        star_info['blood_type'] = dd_text
                    elif dt_text == '身高':
                        height_match = re.search(r'(\d+(?:\.\d+)?)', dd_text)
                        star_info['height'] = height_match.group(1) if height_match else dd_text
                    elif dt_text == '体重':
                        weight_match = re.search(r'(\d+(?:\.\d+)?)', dd_text)
                        star_info['weight'] = weight_match.group(1) if weight_match else dd_text
                    elif dt_text in ['出生日期', '出生年月']:
                        year_match = re.search(r'(\d{4})年', dd_text)
                        star_info['birth_year'] = year_match.group(1) if year_match else dd_text
                print(f"✅ 成功提取{name}基本信息")
            else:
                print(f"⚠️ 未找到{name}的基本信息模块")

            # 爬取图片（修复startswith报错：先判断src是字符串）
            pic_urls = []
            # 入口1：摘要区图片
            summary_imgs = bs.select('.summary-pic img, .summary-content img')
            for img in summary_imgs:
                src = img.get('src')
                # 先判断src是字符串且不为空，再调用startswith（解决“None无startswith”）
                if isinstance(src, str) and src.startswith(('http://', 'https://')):
                    pic_urls.append(src)
            
            # 入口2：更多图片链接（修复find参数报警：正则对象合法，无需修改）
            more_pic_a = bs.find('a', text=re.compile(r'更多图片')) or bs.select_one('.summary-pic a[href*="/item/pic/"]')
            if more_pic_a and more_pic_a.get('href'):
                pic_list_url = f'https://baike.baidu.com{clean_baike_url(more_pic_a.get("href"))}'
                time.sleep(random.uniform(0.5, 1))
                pic_response = session.get(pic_list_url, headers=get_random_headers(), timeout=20)
                pic_response.raise_for_status()
                pic_bs = BeautifulSoup(pic_response.text, 'lxml')
                # 入口2图片：同样判断src类型
                list_imgs = pic_bs.select('.pic-list img, .album-list img')
                for img in list_imgs:
                    src = img.get('src')
                    if isinstance(src, str) and src.startswith(('http://', 'https://')):
                        pic_urls.append(src)

            # 去重并下载（修复down_save_pic参数不匹配）
            unique_pic_urls = list(set(pic_urls))
            if unique_pic_urls:
                print(f"✅ 找到{name}的{len(unique_pic_urls)}张图片，开始下载")
                down_save_pic(name, unique_pic_urls, pic_root_dir)  # 传3个参数，对应函数定义
            else:
                print(f"⚠️ 未找到{name}的有效图片链接")

            star_infos.append(star_info)
            print(f"✅ {name}爬取完成")

        except requests.exceptions.HTTPError as e:
            if '403' in str(e):
                err_msg = f"❌ {name}爬取失败：403反爬拦截（建议添加Cookie）"
            else:
                err_msg = f"❌ {name}爬取失败：HTTP错误（{str(e)}）"
            print(err_msg)
            star_infos.append(star_info)
        except Exception as e:
            print(f"❌ {name}爬取失败：其他错误（{str(e)}）")
            star_infos.append(star_info)
            continue

    # 保存信息
    with open('work/stars_info.json', 'w', encoding='UTF-8') as f:
        json.dump(star_infos, f, ensure_ascii=False, indent=2)
    print(f"\n=== 所有爬取任务结束 ===")
    print(f"共处理{len(valid_stars)}位嘉宾，保存至work/stars_info.json")
    print(f"图片保存目录：{os.path.abspath(pic_root_dir)}")


def down_save_pic(name, pic_urls, pic_root_dir):
    """修改函数定义，接受3个参数（解决“应为2个位置参数”报错）"""
    # 基于pic_root_dir生成图片保存路径
    pic_dir = os.path.join(pic_root_dir, name)
    os.makedirs(pic_dir, exist_ok=True)

    for i, pic_url in enumerate(pic_urls, start=1):
        try:
            pic_response = requests.get(pic_url, timeout=15, stream=True)
            pic_response.raise_for_status()
            # 验证图片格式
            if 'image' not in pic_response.headers.get('Content-Type', ''):
                print(f"跳过非图片链接：{pic_url}")
                continue
            # 保存图片
            pic_path = os.path.join(pic_dir, f'{i}.jpg')
            with open(pic_path, 'wb') as f:
                for chunk in pic_response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"下载{name}的第{i}张图片失败：{str(e)}")
            continue


if __name__ == '__main__':
    os.makedirs('work', exist_ok=True)
    wiki_table = crawl_wiki_data()
    parse_wiki_data(wiki_table)
    crawl_everyone_wiki_urls()
    print("\n=== 所有爬取任务结束 ===")