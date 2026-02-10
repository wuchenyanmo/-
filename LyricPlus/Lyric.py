import time
import re

RE_METADATA = re.compile(r'^\[([a-zA-Z]+):(.*)\]$')
RE_LYRICS = re.compile(r'^\[(\d{2,}:\d{2}\.\d{2,3})\](.*)$')

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
                
    def to_lrc(self) -> str:
        lrc = []
        if(self.metadata_keys):
            for key in self.metadata_keys:
                lrc.append(f"[{key}:{self.metadata[key]}]")
        for timestamp, text in zip(self.timestamps, self.lyrics):
            minute = int(timestamp // 60)
            second = timestamp % 60
            lrc.append(f"[{minute:02d}:{second:05.2f}]{text}")
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
    