import time
import re
from collections import Counter

RE_METADATA = re.compile(r'^\[([a-zA-Z]+):(.*)\]$')
RE_LYRICS = re.compile(r'^\[(\d{2,}:\d{2}\.\d{2,3})\](.*)$')

TRANSLATION_BRACKETS = {'{}', '｛｝', '[]', '［］', '()', '（）', 
                        '「」', '『』', '【】', '〖〗', '〔〕'}
class LyricLineStamp:
    '''
    带行级时间戳的歌词对象
    '''
    def __init__(self, lrc: str):
        self.metadata = {}
        # 为了保持原有元数据的顺序，将顺序储存于列表
        self.metadata_keys = []
        self.timestamps = []
        self.lyrics = []
        for line in lrc.splitlines():
            line = line.strip()
            meta_match = RE_METADATA.match(line)
            if meta_match:
                key, value = meta_match.groups()
                self.metadata[key] = value
                self.metadata_keys.append(key)
                continue
            lyric_match = RE_LYRICS.match(line)
            if lyric_match:
                timestamp, text = lyric_match.groups()
                minute, second = timestamp.split(':')
                self.timestamps.append(60 * float(minute) + float(second))
                self.lyrics.append(text)
                
    def to_lrc(self, translation: bool = False, brackets: str = "【】") -> str:
        lrc = []
        if(translation and not hasattr(self, "translation")):
            raise ValueError("It should load translation first")
        if(self.metadata_keys):
            for key in self.metadata_keys:
                lrc.append(f"[{key}:{self.metadata[key]}]")
        for i, (timestamp, text) in enumerate(zip(self.timestamps, self.lyrics)):
            minute = int(timestamp // 60)
            second = timestamp % 60
            line = f"[{minute:02d}:{second:05.2f}]{text}"
            if(translation and self.translation[i]):
                line = line + f"{brackets[0]}{self.translation[i]}{brackets[1]}"
            lrc.append(line)
        return '\n'.join(lrc)
    
    def load_translation(self, translation_lrc: str):
        """
        加载翻译歌词，根据最近时间戳对齐到原歌词，忽略元数据
        如果不是一一匹配，后一句可能会覆盖前一句
        """
        self.translation = ['' for _ in self.lyrics]
        for line in translation_lrc.splitlines():
            line = line.strip()
            lyric_match = RE_LYRICS.match(line)
            if lyric_match:
                timestamp_str, text = lyric_match.groups()
                minute, second = timestamp_str.split(':')
                trans_timestamp = 60 * float(minute) + float(second)
                closest_idx = min(
                    range(len(self.timestamps)), 
                    key=lambda i: abs(self.timestamps[i] - trans_timestamp)
                )
                self.translation[closest_idx] = text
                
    def clean_translation(self, brackets: set[str] = TRANSLATION_BRACKETS,
                          threshold: float = 0.8, only_detect: bool = False) -> str | None:
        '''
        去除翻译歌词两端可能存在的括号，括号由brackets参数定义，返回侦测到的括号对，若无，返回None
        
        brackets: 集合，定义了可能的括号，每个元素形如'【】'
        
        threshold: 如果超过该阈值的翻译文本头尾含有同一个括号，则认为翻译文本被该括号括起
        
        only_detect: 若为真，则仅侦测而不去除括号
        '''
        if(not hasattr(self, "translation")):
            raise ValueError("It should load translation first")
        trans_texts = [t for t in self.translation if len(t) > 1]
        headtails = [(t[0] + t[-1]) for t in trans_texts]
        headtails_count = Counter(headtails)
        headtails_first, freq = headtails_count.most_common(1)[0]
        if((freq < threshold * len(trans_texts)) or (headtails_first not in brackets)):
            return None
        if(only_detect):
            return headtails_first
        
        re_bracket = re.compile(rf"^{headtails_first[0]}(.*){headtails_first[1]}$")
        for i in range(len(self.translation)):
            bracket_match = re_bracket.match(self.translation[i])
            if(bracket_match):
                self.translation[i] = bracket_match.group(1)
        return headtails_first