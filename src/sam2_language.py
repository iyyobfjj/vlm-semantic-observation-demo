from __future__ import annotations


LABEL_NAMES_ZH = {
    "storage_cabinet": "储物柜",
    "switch": "开关",
    "refrigerator": "冰箱",
    "microwave": "微波炉",
    "rubbish_bin": "垃圾桶",
}

POSITION_NOTE_ZH = "位置为图像二维方位，不代表地图坐标或真实距离。"

POSITION_ORDER = {
    "左上方": 0,
    "左侧": 1,
    "左下方": 2,
    "上方": 3,
    "中央": 4,
    "下方": 5,
    "右上方": 6,
    "右侧": 7,
    "右下方": 8,
}


def label_name_zh(label: str) -> str:
    return LABEL_NAMES_ZH.get(label, label.replace("_", " "))


def image_position_zh(center_normalized: tuple[float, float]) -> str:
    x, y = center_normalized
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError(f"normalized center must be inside the image: {center_normalized}")

    horizontal = 0 if x < 1.0 / 3.0 else 1 if x < 2.0 / 3.0 else 2
    vertical = 0 if y < 1.0 / 3.0 else 1 if y < 2.0 / 3.0 else 2
    positions = (
        ("左上方", "上方", "右上方"),
        ("左侧", "中央", "右侧"),
        ("左下方", "下方", "右下方"),
    )
    return positions[vertical][horizontal]


def join_chinese_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return "、".join(phrases[:-1]) + "和" + phrases[-1]
