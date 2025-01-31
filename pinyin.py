import pinyin

def text_to_pinyin(text):
    # 转换中文文本为拼音列表，包括声调信息
    return [pinyin.lazy_pinyin(word, with_tone_marks=True) for word in text]

# 示例中文文本
chinese_text = "你好，我是示例文本。"
pinyin_sequence = text_to_pinyin(chinese_text)