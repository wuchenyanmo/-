import re
import unicodedata
from collections import Counter

RE_METADATA = re.compile(r'^\[([a-zA-Z]+):(.*)\]$')
RE_LYRICS = re.compile(r'^\[(\d{2,}:\d{2}\.\d{2,3})\](.*)$')
RE_SPACES = re.compile(r'\s+')
RE_FURIGANA = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff々〆ヵヶ]+)\(([\u3040-\u30ffー・ ]+)\)')
RE_METADATA_SEP = re.compile(r'\s*[:：|｜]\s*')
RE_SINGER_COLON = re.compile(r'^(?P<label>[^\s:：|｜]{1,24})\s*[:：|｜]\s*(?P<body>.+)$')
RE_SINGER_BRACKET = re.compile(
    r'^[\(\[（【「『<＜](?P<label>.{1,24}?)[\)\]）】」』>＞]\s*(?P<body>.+)$'
)
RE_SINGER_DASH = re.compile(
    r'^(?P<label>[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff/&+・ ]{1,24})\s*[-~]\s*(?P<body>.+)$'
)

TRANSLATION_BRACKETS = {
    '{}', '｛｝', '[]', '［］', '()', '（）', '「」', '『』', '【】', '〖〗', '〔〕'
}
DEFAULT_METADATA_KEYS = {
    'ti', 'ar', 'al', 'by', 'offset', 'la', 'id', 'length', 're', 've',
}
METADATA_LABELS = {
    '作词', '作詞', '词', '詞', '作曲', '曲', '编曲', '編曲', '作词作曲', '词曲',
    '演唱', '歌手', '原唱', 'vocal', 'artist', 'title', '标题', '標題', 'album',
    '专辑', '專輯', '制作', '製作', 'producer', 'lyrics', 'music', 'composer',
    'lyricist', 'chorus', '和声', '和聲', 'mix', 'mixed', 'master', '录音', '錄音',
}
SINGER_ROLE_WORDS = {
    '男', '女', '合', '和', '齐', '齊', '全', '全员', '全員', 'duet', 'solo',
    'chorus', 'rap', 'lead', 'main', 'vocal', '和声', '和聲',
}


class LyricTokenLine:
    '''
    歌词解析后的行对象
    '''
    def __init__(
        self,
        timestamp: float,
        raw_text: str,
        text: str,
        normalized_text: str,
        is_metadata: bool,
        singer: str | None = None,
        translation: str = '',
        metadata_key: str | None = None,
        metadata_value: str | None = None,
    ):
        self.timestamp = timestamp
        self.raw_text = raw_text
        self.text = text
        self.normalized_text = normalized_text
        self.is_metadata = is_metadata
        self.singer = singer
        self.translation = translation
        self.metadata_key = metadata_key
        self.metadata_value = metadata_value


class LyricLineStamp:
    '''
    带行级时间戳的歌词对象
    '''
    def __init__(self, lrc: str):
        self.metadata = {}
        self.metadata_keys = []
        self.line_infos = []

        for line in lrc.splitlines():
            line = line.strip()
            if not line:
                continue

            meta_match = RE_METADATA.match(line)
            if meta_match:
                key, value = meta_match.groups()
                self._add_metadata(key, value)
                continue

            lyric_match = RE_LYRICS.match(line)
            if not lyric_match:
                continue

            timestamp, text = lyric_match.groups()
            minute, second = timestamp.split(':')
            parsed = self._parse_lyric_line(60 * float(minute) + float(second), text)
            self.line_infos.append(parsed)

            if parsed.is_metadata:
                self._add_metadata(parsed.metadata_key or 'meta', parsed.metadata_value or parsed.text)
                continue

    @property
    def lyric_lines(self) -> list[LyricTokenLine]:
        return [line for line in self.line_infos if not line.is_metadata]

    @property
    def timestamps(self) -> list[float]:
        return [line.timestamp for line in self.lyric_lines]

    @property
    def lyrics(self) -> list[str]:
        return [line.text for line in self.lyric_lines]

    @property
    def normalized_lyrics(self) -> list[str]:
        return [line.normalized_text for line in self.lyric_lines]

    @property
    def singers(self) -> list[str | None]:
        return [line.singer for line in self.lyric_lines]

    @property
    def translation(self) -> list[str]:
        return [line.translation for line in self.lyric_lines]

    def _add_metadata(self, key: str, value: str):
        key_norm = self.normalize_plain_text(key, keep_spaces=False).lower()
        value = self.normalize_plain_text(value)
        if key_norm not in self.metadata:
            self.metadata_keys.append(key_norm)
        self.metadata[key_norm] = value

    @staticmethod
    def normalize_plain_text(text: str, keep_spaces: bool = True) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('’', "'").replace('`', "'")
        text = text.replace('“', '"').replace('”', '"')
        text = text.strip()
        if keep_spaces:
            text = RE_SPACES.sub(' ', text)
        else:
            text = ''.join(text.split())
        return text

    @classmethod
    def normalize_lyric_text(cls, text: str) -> str:
        text = cls.normalize_plain_text(text)
        text = RE_FURIGANA.sub(r'\1', text)
        text = re.sub(r'[·•♪♬♫]+', ' ', text)
        text = re.sub(r'[，。！？、,.!?;；:：~～…]+', ' ', text)
        text = RE_SPACES.sub(' ', text).strip()
        return text

    @classmethod
    def _looks_like_metadata_label(cls, label: str) -> bool:
        normalized = cls.normalize_plain_text(label, keep_spaces=False).lower()
        metadata_words = {item.lower() for item in METADATA_LABELS}
        return normalized in DEFAULT_METADATA_KEYS or normalized in metadata_words

    @classmethod
    def _split_key_value(cls, text: str) -> tuple[str, str] | None:
        parts = RE_METADATA_SEP.split(text, maxsplit=1)
        if len(parts) != 2:
            return None
        key, value = parts[0].strip(), parts[1].strip()
        if not key or not value:
            return None
        return key, value

    @classmethod
    def _metadata_score(cls, timestamp: float, raw_text: str) -> tuple[int, str | None, str | None]:
        text = cls.normalize_plain_text(raw_text)
        compact = cls.normalize_plain_text(raw_text, keep_spaces=False)
        if not text:
            return 0, None, None

        score = 0
        metadata_key = None
        metadata_value = None

        key_value = cls._split_key_value(text)
        if key_value is not None:
            key, value = key_value
            if cls._looks_like_metadata_label(key):
                score += 4
                metadata_key, metadata_value = key, value
            elif timestamp <= 20 and len(key) <= 12 and len(value) <= 40:
                score += 2
                metadata_key, metadata_value = key, value

        if timestamp <= 20:
            score += 1
        if len(text) <= 24:
            score += 1
        if 'instrumental' in compact.lower() or 'inst' in compact.lower():
            score += 3

        separator_count = sum(ch in ':：|｜/' for ch in text)
        lyric_char_count = sum(
            ch.isalpha() or '\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff'
            for ch in text
        )
        if separator_count >= 1 and lyric_char_count <= 20:
            score += 1

        return score, metadata_key, metadata_value

    @classmethod
    def _looks_like_singer_label(cls, label: str) -> bool:
        clean = cls.normalize_plain_text(label)
        compact = cls.normalize_plain_text(label, keep_spaces=False)
        if not clean or len(clean) > 24:
            return False
        if cls._looks_like_metadata_label(clean):
            return False
        if compact.lower() in {item.lower() for item in SINGER_ROLE_WORDS}:
            return True
        if re.fullmatch(r'[A-Za-z0-9/&+]+', compact):
            return True
        if re.fullmatch(r'[\u3040-\u30ff\u3400-\u9fffA-Za-z0-9/&+・ ]+', clean):
            return True
        return False

    @classmethod
    def _extract_singer_marker(cls, text: str) -> tuple[str | None, str]:
        for pattern in (RE_SINGER_BRACKET, RE_SINGER_COLON, RE_SINGER_DASH):
            match = pattern.match(text)
            if not match:
                continue
            label = cls.normalize_plain_text(match.group('label'))
            body = match.group('body').strip()
            if body and cls._looks_like_singer_label(label):
                return label, body
        return None, text

    @classmethod
    def _parse_lyric_line(cls, timestamp: float, raw_text: str) -> LyricTokenLine:
        text = cls.normalize_plain_text(raw_text)
        score, metadata_key, metadata_value = cls._metadata_score(timestamp, text)
        if score >= 4:
            return LyricTokenLine(
                timestamp=timestamp,
                raw_text=raw_text,
                text=text,
                normalized_text='',
                is_metadata=True,
                metadata_key=metadata_key,
                metadata_value=metadata_value or text,
            )

        singer, body = cls._extract_singer_marker(text)
        normalized_text = cls.normalize_lyric_text(body)
        if not normalized_text and score >= 2:
            return LyricTokenLine(
                timestamp=timestamp,
                raw_text=raw_text,
                text=text,
                normalized_text='',
                is_metadata=True,
                metadata_key=metadata_key,
                metadata_value=metadata_value or text,
            )

        return LyricTokenLine(
            timestamp=timestamp,
            raw_text=raw_text,
            text=body,
            normalized_text=normalized_text,
            is_metadata=False,
            singer=singer,
        )

    def get_alignment_texts(self, drop_empty: bool = True) -> list[str]:
        '''
        获取适用于对齐的规范化歌词文本
        '''
        if drop_empty:
            return [line.normalized_text for line in self.lyric_lines if line.normalized_text]
        return self.normalized_lyrics.copy()

    def to_lrc(self, translation: bool = False, brackets: str = "【】") -> str:
        lrc = []
        if(translation and not hasattr(self, "translation")):
            raise ValueError("It should load translation first")
        if(self.metadata_keys):
            for key in self.metadata_keys:
                lrc.append(f"[{key}:{self.metadata[key]}]")
        for line in self.lyric_lines:
            timestamp = line.timestamp
            text = line.text
            minute = int(timestamp // 60)
            second = timestamp % 60
            lrc_line = f"[{minute:02d}:{second:05.2f}]{text}"
            if(translation and line.translation):
                lrc_line = lrc_line + f"{brackets[0]}{line.translation}{brackets[1]}"
            lrc.append(lrc_line)
        return '\n'.join(lrc)

    def load_translation(self, translation_lrc: str):
        """
        加载翻译歌词，根据最近时间戳对齐到原歌词，忽略元数据
        如果不是一一匹配，后一句可能会覆盖前一句
        """
        lyric_lines = self.lyric_lines
        for line_info in lyric_lines:
            line_info.translation = ''
        for line in translation_lrc.splitlines():
            line = line.strip()
            lyric_match = RE_LYRICS.match(line)
            if lyric_match:
                timestamp_str, text = lyric_match.groups()
                minute, second = timestamp_str.split(':')
                trans_timestamp = 60 * float(minute) + float(second)
                closest_idx = min(
                    range(len(lyric_lines)),
                    key=lambda i: abs(lyric_lines[i].timestamp - trans_timestamp)
                )
                lyric_lines[closest_idx].translation = text

    def clean_translation(self, brackets: set[str] = TRANSLATION_BRACKETS,
                          threshold: float = 0.8, only_detect: bool = False) -> str | None:
        '''
        去除翻译歌词两端可能存在的括号，括号由brackets参数定义，返回侦测到的括号对，若无，返回None

        brackets: 集合，定义了可能的括号，每个元素形如'【】'

        threshold: 如果超过该阈值的翻译文本头尾含有同一个括号，则认为翻译文本被该括号括起

        only_detect: 若为真，则仅侦测而不去除括号
        '''
        trans_texts = [line.translation for line in self.lyric_lines if len(line.translation) > 1]
        if not trans_texts:
            return None
        headtails = [(t[0] + t[-1]) for t in trans_texts]
        headtails_count = Counter(headtails)
        headtails_first, freq = headtails_count.most_common(1)[0]
        if((freq < threshold * len(trans_texts)) or (headtails_first not in brackets)):
            return None
        if(only_detect):
            return headtails_first

        re_bracket = re.compile(rf"^{re.escape(headtails_first[0])}(.*){re.escape(headtails_first[1])}$")
        for line in self.lyric_lines:
            bracket_match = re_bracket.match(line.translation)
            if(bracket_match):
                line.translation = bracket_match.group(1)
        return headtails_first
