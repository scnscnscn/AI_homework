import json
import requests
from bs4 import BeautifulSoup
import os
import re

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
        tables = soup.find_all('table')  # 获取所有表格

        # 1. 调整标题关键词：匹配可能的嘉宾列表标题（根据当前页面调整）
        target_keywords = ['嘉宾', '成员', '参赛', '选手']  # 更通用的关键词

        for table in tables:
            # 2. 扩大前置标签范围：查找表格前的所有可能标题标签（h3、div、p等）
            # 向前查找最近的标题标签（优先h3，其次div，最后p）
            title_tags = table.find_previous(['h3', 'div', 'p'])
            if not title_tags:
                continue  # 无前置标题，跳过

            # 3. 清洗标题文本（去空格、小写化），提高匹配容错性
            title_text = title_tags.get_text(strip=True).lower()
            # 判断标题是否包含目标关键词
            if any(keyword in title_text for keyword in target_keywords):
                # 4. 额外验证表格是否包含嘉宾链接（避免误判）
                if table.find('a', href=re.compile(r'/item/')):  # 包含指向个人百科的链接
                    print(f"找到目标表格，标题包含关键词：{title_text}")
                    return table

        # 如果未通过标题匹配，尝试直接通过表格特征定位（备选方案）
        for table in tables:
            # 特征：表格包含多个<a>标签，且href指向百度百科条目（/item/xxx）
            a_tags = table.find_all('a', href=re.compile(r'/item/'))
            if len(a_tags) > 5:  # 假设嘉宾数量大于5
                print("通过表格特征找到疑似嘉宾表格")
                return table

        print("未找到嘉宾信息表格（页面结构可能已变更）")
        return None

    except Exception as e:
        print(f"爬取百科表格失败：{str(e)}")
        return None


def parse_wiki_data(table_html):
    """解析嘉宾表格，提取姓名和百科链接，保存为stars.json（需先判断table_html非None）"""
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
            # 3. 先判断是否找到a标签，避免None访问属性
            a_tag = td.find_next('a')
            if a_tag is not None:
                star["name"] = a_tag.get_text(strip=True, separator='')  # 处理多换行/空格文本
                href = a_tag.get('href')
                # 处理href可能为None的情况（部分a标签无href）
                star['link'] = f'https://baike.baidu.com{href}' if href else ''
                stars.append(star)
                continue  # 找到a标签则跳过后续div判断
            
            # 若未找到直接a标签，尝试找div下的a标签
            div_tag = td.find_next('div')
            if div_tag is not None:
                div_a_tag = div_tag.find('a')
                if div_a_tag is not None:
                    star["name"] = div_a_tag.get_text(strip=True, separator='')
                    href = div_a_tag.get('href')
                    star['link'] = f'https://baike.baidu.com{href}' if href else ''
                    stars.append(star)

    # 4. 优化JSON保存：直接dump列表，无需转字符串再解析（原写法不规范）
    os.makedirs('work', exist_ok=True)  # 确保work目录存在
    with open('work/stars.json', 'w', encoding='UTF-8') as f:
        json.dump(stars, f, ensure_ascii=False, indent=2)  # indent=2美化格式
    print(f"已解析{len(stars)}位嘉宾信息，保存至stars.json")


def crawl_everyone_wiki_urls():
    """爬取每位嘉宾的详细信息和图片，保存stars_info.json和图片"""
    # 先判断stars.json是否存在
    star_json_path = 'work/stars.json'
    if not os.path.exists(star_json_path):
        print(f"{star_json_path}不存在，无法爬取嘉宾详情")
        return

    with open(star_json_path, 'r', encoding='UTF-8') as file:
        json_array = json.load(file)
        # 过滤无效数据（无name或link的条目）
        json_array = [star for star in json_array if star.get('name') and star.get('link')]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }
    star_infos = []  # 统一收集所有嘉宾信息，避免循环中重复写文件

    for star in json_array:
        name = star['name']
        link = star['link']
        star_info = {'name': name, 'link': link}  # 保留原始链接，便于排查
        print(f"\n开始爬取嘉宾：{name}")

        try:
            # 爬取嘉宾个人百科页面
            response = requests.get(link, headers=headers, timeout=15)
            response.raise_for_status()
            bs = BeautifulSoup(response.text, 'lxml')

            # 5. 先判断基本信息div是否存在，避免None访问find_all
            base_info_div = bs.find('div', {'class': 'basic-info J-basic-info cmn-clearfix'})
            if base_info_div is not None:
                dls = base_info_div.find_all('dl')
                for dl in dls:
                    dts = dl.find_all('dt')
                    for dt in dts:
                        dt_text = dt.get_text(strip=True).replace('：', '')  # 处理“民族：”这类文本
                        # 6. 先判断dd是否存在，避免None访问text
                        dd_tag = dt.find_next('dd')
                        if dd_tag is None:
                            continue
                        dd_text = dd_tag.get_text(strip=True, separator=' ')  # 合并多行为空格分隔

                        # 提取目标字段，处理特殊格式（如身高、出生日期）
                        if dt_text == '民族':
                            star_info['nation'] = dd_text
                        elif dt_text == '星座':
                            star_info['constellation'] = dd_text
                        elif dt_text == '血型':
                            star_info['blood_type'] = dd_text
                        elif dt_text == '身高':
                            # 提取cm前的数字（如“165cm”→“165”）
                            height_match = re.search(r'(\d+(?:\.\d+)?)cm', dd_text)
                            star_info['height'] = height_match.group(1) if height_match else dd_text
                        elif dt_text == '体重':
                            # 提取kg前的数字（如“48kg”→“48”）
                            weight_match = re.search(r'(\d+(?:\.\d+)?)kg', dd_text)
                            star_info['weight'] = weight_match.group(1) if weight_match else dd_text
                        elif dt_text == '出生日期':
                            # 提取年份（如“1985年10月15日”→“1985”）
                            year_match = re.search(r'(\d{4})年', dd_text)
                            star_info['birth_year'] = year_match.group(1) if year_match else dd_text
            else:
                print(f"未找到{name}的基本信息模块")

            # 爬取嘉宾图片列表
            # 7. 判断图片列表链接是否存在
            pic_list_a = bs.select_one('.summary-pic a')  # select_one返回单个元素（无则None）
            if pic_list_a is not None:
                pic_list_href = pic_list_a.get('href')
                if pic_list_href:
                    pic_list_url = f'https://baike.baidu.com{pic_list_href}'
                    # 爬取图片列表页面
                    pic_response = requests.get(pic_list_url, headers=headers, timeout=15)
                    pic_response.raise_for_status()
                    pic_bs = BeautifulSoup(pic_response.text, 'lxml')
                    pic_imgs = pic_bs.select('.pic-list img')
                    pic_urls = [img.get('src') for img in pic_imgs if img.get('src')]  # 过滤None链接
                    print(f"找到{name}的{len(pic_urls)}张图片，开始下载")
                    down_save_pic(name, pic_urls)
                else:
                    print(f"{name}的图片列表链接为空")
            else:
                print(f"未找到{name}的图片列表入口")

            star_infos.append(star_info)

        except Exception as e:
            print(f"爬取{name}信息失败：{str(e)}")
            continue  # 某嘉宾失败不影响其他

    # 8. 统一保存所有嘉宾信息（避免循环中重复写文件）
    with open('work/stars_info.json', 'w', encoding='UTF-8') as f:
        json.dump(star_infos, f, ensure_ascii=False, indent=2)
    print(f"\n所有嘉宾信息爬取完成，共{len(star_infos)}位嘉宾，保存至stars_info.json")


def down_save_pic(name, pic_urls):
    """下载图片到work/pics/name目录，处理目录创建和下载异常"""
    # 9. 优化路径处理：使用os.path.join避免跨平台问题
    pic_dir = os.path.join('work', 'pics', name)
    os.makedirs(pic_dir, exist_ok=True)  # exist_ok=True避免目录已存在报错

    for i, pic_url in enumerate(pic_urls, start=1):  # start=1使图片编号从1开始
        try:
            # 下载图片（设置timeout，避免长时间阻塞）
            pic_response = requests.get(pic_url, timeout=15, stream=True)  # stream=True适合大文件
            pic_response.raise_for_status()
            # 验证图片格式（简单判断Content-Type）
            if 'image' not in pic_response.headers.get('Content-Type', ''):
                print(f"跳过非图片链接：{pic_url}")
                continue
            # 保存图片
            pic_path = os.path.join(pic_dir, f'{i}.jpg')
            with open(pic_path, 'wb') as f:
                for chunk in pic_response.iter_content(chunk_size=1024):  # 分块写入（避免内存占用过大）
                    if chunk:
                        f.write(chunk)
            # print(f"成功下载{name}的第{i}张图片")
        except Exception as e:
            print(f"下载{name}的第{i}张图片失败：{str(e)}")
            continue


if __name__ == '__main__':
    # 确保work目录存在（避免首次运行时创建失败）
    os.makedirs('work', exist_ok=True)
    # 1. 爬取嘉宾表格
    wiki_table = crawl_wiki_data()
    # 2. 解析表格并保存姓名+链接
    parse_wiki_data(wiki_table)
    # 3. 爬取每位嘉宾的详细信息和图片
    crawl_everyone_wiki_urls()
    print("\n=== 所有爬取任务结束 ===")