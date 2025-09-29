import json
import requests
from bs4 import BeautifulSoup
import os
import re
import time
import random


# -------------------------- 原有辅助函数（无修改） --------------------------
def clean_baike_url(url):
    if isinstance(url, str) and '?' in url:
        url = url.split('?')[0]
    return url


def get_random_headers():
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


# -------------------------- 原有函数（无修改） --------------------------
def crawl_wiki_data():
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


# -------------------------- 修改后：仅保留嘉宾基本信息爬取 --------------------------
def crawl_everyone_wiki_urls():
    star_json_path = 'work/stars.json'
    if not os.path.exists(star_json_path):
        print(f"{star_json_path}不存在，无法爬取嘉宾详情")
        return

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
            headers = get_random_headers()
            response = session.get(link, headers=headers, timeout=20)
            response.raise_for_status()
            bs = BeautifulSoup(response.text, 'lxml')

            # 基本信息提取
            base_info_div = bs.find('div', class_='basicInfo_rZDFN J-basic-info')
            if base_info_div:
                item_wrappers = base_info_div.find_all('div', class_='itemWrapper_u4OET')
                for wrapper in item_wrappers:
                    dt_tag = wrapper.find('dt', class_='basicInfoItem_TsAAR itemName_tPWxP')
                    dd_tag = wrapper.find('dd', class_='basicInfoItem_TsAAR itemValue_Xsiqq')
                    if not (dt_tag and dd_tag):
                        continue

                    dt_text = dt_tag.get_text(strip=True)
                    dt_text = dt_text.replace('\u00A0', '').replace('：', '')
                    value_span = dd_tag.find('span', class_='text_zBf3n')
                    dd_text = value_span.get_text(strip=True, separator=' ') if value_span else dd_tag.get_text(strip=True, separator=' ')

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


# -------------------------- 入口函数（无修改） --------------------------
if __name__ == '__main__':
    os.makedirs('work', exist_ok=True)
    wiki_table = crawl_wiki_data()
    parse_wiki_data(wiki_table)
    crawl_everyone_wiki_urls()
    print("\n=== 所有爬取任务结束 ===")