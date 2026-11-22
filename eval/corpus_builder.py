#!/usr/bin/env python3
"""Deterministic, license-clean multi-domain corpus and gold boundary generator for kototok."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "artifacts" / "data"

# Multiplier on train-corpus byte targets and generator pool sizes.
# Test-split byte targets and the gold set stay fixed for comparability.
TRAIN_SCALE = float(os.environ.get("KOTOTOK_TRAIN_SCALE", "1.0"))

# ---------------------------------------------------------------------------
# Lexicons for Japanese multi-genre generation (Tech, Encyclopedia, News, Literary)
# ---------------------------------------------------------------------------

JA_TECH_SUBJECTS = [
    "トークナイザー", "アルゴリズム", "コンパイラ", "データベース", "アーキテクチャ",
    "インターフェース", "パラメータ", "ネットワーク", "サーバー", "フレームワーク",
    "モジュール", "エンコーダ", "デコーダ", "スケジューラ", "プロファイラ",
    "パイプライン", "キャッシュ", "ランタイム", "コンテナ", "ストレージ",
    "自然言語処理", "形態素解析", "文字種遷移", "語彙最適化", "並列計算",
    "分散システム", "情報検索", "機械翻訳", "音声認識", "知識表現",
    "データ構造", "木構造", "配列操作", "メモリ管理", "設計指針",
    "高速化手法", "評価指標", "言語モデル", "圧縮効率", "境界精度",
]

JA_TECH_OBJECTS = [
    "バイナリ列", "バイト配列", "語彙表", "状態遷移表", "文字クラス",
    "トークン列", "中間表現", "学習コーパス", "実験結果", "計算量",
    "入力文字列", "補助記憶", "設定ファイル", "標準規格", "統計的手法",
    "処理時間", "スループット", "レイテンシ", "オーバーヘッド", "バッファ",
    "ポインタ", "イテレータ", "スレッド", "ハッシュ表", "優先度付きキュー",
    "文脈長", "サブワード", "ユニグラム", "ベースライン", "ハイパーパラメータ",
]

JA_VERBS_OKURIGANA = [
    ("調べる", "調べた", "調べて", "調べている"),
    ("作る", "作った", "作って", "作っている"),
    ("動く", "動いた", "動いて", "動いている"),
    ("考える", "考えた", "考えて", "考えている"),
    ("確かめる", "確かめた", "確かめて", "確かめている"),
    ("書き換える", "書き換えた", "書き換えて", "書き換えている"),
    ("切り替える", "切り替えた", "切り替えて", "切り替えている"),
    ("学ぶ", "学んだ", "学んで", "学んでいる"),
    ("読む", "読んだ", "読んで", "読んでいる"),
    ("書く", "書いた", "書いて", "書いている"),
    ("見る", "見た", "見て", "見ている"),
    ("聞く", "聞いた", "聞いて", "聞いている"),
    ("話す", "話した", "話して", "話している"),
    ("始める", "始めた", "始めて", "始めている"),
    ("終わる", "終わった", "終わって", "終わっている"),
    ("変わる", "変わった", "変わって", "変わっている"),
    ("残る", "残った", "残って", "残っている"),
    ("増やす", "増やした", "増やして", "増やしている"),
    ("減らす", "減らした", "減らして", "減らしている"),
    ("進める", "進めた", "進めて", "進めている"),
    ("導く", "導いた", "導いて", "導いている"),
    ("支える", "支えた", "支えて", "支えている"),
    ("抑える", "抑えた", "抑えて", "抑えている"),
    ("見直す", "見直した", "見直して", "見直している"),
    ("受け取る", "受け取った", "受け取って", "受け取っている"),
    ("組み立てる", "組み立てた", "組み立てて", "組み立てている"),
    ("取り扱う", "取り扱った", "取り扱って", "取り扱っている"),
    ("見つける", "見つけた", "見つけて", "見つけている"),
    ("比べる", "比べた", "比べて", "比べている"),
    ("続ける", "続けた", "続けて", "続けている"),
    ("試みる", "試みた", "試みて", "試みている"),
    ("求める", "求めた", "求めて", "求めている"),
    ("保つ", "保った", "保って", "保っている"),
    ("測る", "測った", "測って", "測っている"),
    ("示す", "示した", "示して", "示している"),
    ("表す", "表した", "表して", "表している"),
    ("整える", "整えた", "整えて", "整えている"),
    ("補う", "補った", "補って", "補っている"),
    ("避ける", "避けた", "避けて", "避けている"),
    ("扱う", "扱った", "扱って", "扱っている"),
    ("解く", "解いた", "解いて", "解いている"),
]

JA_DAILY_VERBS = [
    ("食べる", "食べた", "食べて", "食べている"),
    ("歩く", "歩いた", "歩いて", "歩いている"),
    ("走る", "走った", "走って", "走っている"),
    ("働く", "働いた", "働いて", "働いている"),
    ("遊ぶ", "遊んだ", "遊んで", "遊んでいる"),
    ("育てる", "育てた", "育てて", "育てている"),
    ("届ける", "届けた", "届けて", "届けている"),
    ("集める", "集めた", "集めて", "集めている"),
    ("選ぶ", "選んだ", "選んで", "選んでいる"),
    ("開く", "開いた", "開いて", "開いている"),
    ("閉じる", "閉じた", "閉じて", "閉じている"),
    ("渡す", "渡した", "渡して", "渡している"),
    ("飾る", "飾った", "飾って", "飾っている"),
    ("描く", "描いた", "描いて", "描いている"),
    ("歌う", "歌った", "歌って", "歌っている"),
    ("登る", "登った", "登って", "登っている"),
    ("泳ぐ", "泳いだ", "泳いで", "泳いでいる"),
    ("休む", "休んだ", "休んで", "休んでいる"),
    ("眠る", "眠った", "眠って", "眠っている"),
    ("笑う", "笑った", "笑って", "笑っている"),
]

JA_ADJECTIVES = [
    ("新しい", "新しく"), ("美しい", "美しく"), ("高い", "高く"), ("低い", "低く"),
    ("速い", "速く"), ("遅い", "遅く"), ("広い", "広く"), ("狭い", "狭く"),
    ("深い", "深く"), ("浅い", "浅く"), ("強い", "強く"), ("弱い", "弱く"),
    ("正しい", "正しく"), ("大きい", "大きく"), ("小さい", "小さく"), ("長い", "長く"),
    ("短い", "短く"), ("明るい", "明るく"), ("暗い", "暗く"), ("古い", "古く"),
    ("難しい", "難しく"), ("易しい", "易しく"), ("詳しい", "詳しく"), ("著しい", "著しく"),
    ("乏しい", "乏しく"), ("珍しい", "珍しく"), ("激しい", "激しく"), ("静かな", "静かに"),
    ("豊かな", "豊かに"), ("確かな", "確かに"), ("穏やかな", "穏やかに"), ("複雑な", "複雑に"),
    ("単純な", "単純に"), ("柔軟な", "柔軟に"), ("正確な", "正確に"), ("安定な", "安定に"),
]

JA_PLACES = [
    "東京", "京都", "大阪", "札幌", "仙台", "名古屋", "福岡", "横浜",
    "神戸", "広島", "金沢", "長野", "筑波", "沖縄", "北海道", "日本",
]

JA_ROLES = [
    "開発者", "研究者", "技術者", "編集者", "設計者", "利用者", "執筆者",
    "観測者", "担当者", "調査員", "専門家", "翻訳家", "学生", "教師",
]

JA_ENCYCLOPEDIA_TOPICS = [
    ("地震観測網", "地下の揺れ", "気象庁", "防災計画"),
    ("海洋生態系", "深海の海流", "水産研究所", "保全活動"),
    ("古代建築", "木造の組物", "文化財団", "修復作業"),
    ("天体望遠鏡", "遠方の銀河", "国立天文台", "画像解析"),
    ("植物図鑑", "高山の花々", "植物園", "標本整理"),
    ("鉄道網", "山間の路線", "交通局", "時刻表改正"),
    ("伝統工芸", "漆器の技法", "工芸館", "展示企画"),
    ("気候変動", "季節の降水", "環境省", "長期予測"),
    ("音楽理論", "和声の構成", "音楽大学", "録音資料"),
    ("都市計画", "街路の緑化", "自治体", "景観整備"),
    ("農業技術", "土壌の水分", "農林試験場", "収穫記録"),
    ("図書館学", "書誌の目録", "中央図書館", "デジタル保存"),
]

JA_LITERARY_NOUNS = [
    "夕暮れの街並み", "朝霧の森", "静かな図書室", "古い石橋", "海辺の灯台",
    "木漏れ日の小道", "冬の星座", "春の風", "秋空の雲", "雨上がりの庭",
    "遠い山の稜線", "川面の波紋", "手紙の余白", "時計台の鐘", "列車窓の景色",
]

JA_FOODS = [
    "温かい味噌汁", "焼きたてのパン", "季節の果物", "炊きたてのご飯",
    "香り高い緑茶", "手作りの和菓子", "新鮮な野菜", "だし巻き卵",
    "冷たい蕎麦", "煮込んだ料理", "濃い珈琲", "甘い林檎",
]

# ---------------------------------------------------------------------------
# English prose lexicons
# ---------------------------------------------------------------------------

EN_SUBJECTS = [
    "The deterministic tokenizer", "A modern compiler pipeline", "Distributed storage",
    "The pre-tokenization automaton", "An asynchronous scheduler", "Memory-mapped indexing",
    "The statistical language model", "Cross-script boundary detection", "Lossless byte encoding",
    "Cache-friendly vector lookup", "A thread-safe priority queue", "The evaluation harness",
    "Byte-level normalization", "Structural code analysis", "Incremental vocabulary pruning",
    "Low-latency stream processing", "The benchmark runner", "Unicode scalar validation",
]

EN_VERBS = [
    "preserves", "eliminates", "accelerates", "verifies", "transforms", "partitions",
    "compresses", "stabilizes", "quantifies", "improves", "reconstructs", "isolates",
    "balances", "normalizes", "optimizes", "tracks", "evaluates", "guarantees",
]

EN_OBJECTS = [
    "worst-case linear runtime", "byte-level round-trip fidelity", "subword compression ratios",
    "UTF-8 boundary invariants", "cross-lingual vocabulary efficiency", "cache locality under load",
    "deterministic token offsets", "memory allocation overhead", "morphological boundary alignment",
    "multi-threaded encoding throughput", "non-overlapping split metrics", "reproducible benchmark results",
]

EN_CLAUSES = [
    "without introducing backtracking stalls on adversarial inputs",
    "while keeping dead vocabulary entries near zero across domains",
    "even when mixing Japanese prose, English documentation, and source code",
    "by enforcing strict invariants over every learned merge event",
    "across held-out evaluation corpora of varying token distributions",
    "under tight 4096-entry and 8192-entry vocabulary budgets",
    "so downstream decoders never encounter malformed byte sequences",
    "during single-threaded latency benchmarks and multi-core runs",
]

EN_ADVERBS = [
    "Consistently,", "In practice,", "By design,", "Under heavy load,",
    "Empirically,", "Across test suites,", "Significantly,", "Notably,",
]

# ---------------------------------------------------------------------------
# Code snippet lexicons (Rust & Python)
# ---------------------------------------------------------------------------

RUST_TYPES = ["TokenBuffer", "ChunkCursor", "VocabSlot", "MergeHeap", "ScriptState", "ByteSpan", "SegmentMap", "RankIndex"]
RUST_FIELDS = ["offset", "capacity", "rank", "score", "chunk_len", "segment_id", "byte_count", "frequency"]
RUST_FUNCS = ["scan_chunks", "apply_merges", "prune_dead", "emit_tokens", "verify_utf8", "split_boundaries", "compute_npmi", "replay_events"]

PY_CLASSES = ["CorpusSample", "BoundaryMetric", "TokenStats", "EvalRecord", "VocabAudit", "SplitConfig", "LatencyProbe", "MergeTrace"]
PY_FUNCS = ["measure_compression", "align_byte_offsets", "calculate_f1", "check_disjoint", "collect_dead_rate", "render_summary", "parse_jsonl", "stream_records"]

# ---------------------------------------------------------------------------
# Curated Gold Japanese Boundary & Okurigana Dataset (150 sentences)
# Each entry: (morphemes, okurigana_indices, noun_particle_pairs)
#   morphemes: list[str] -> exact concatenation == text
#   okurigana_indices: indices in morphemes that are okurigana words
#   noun_particle_pairs: list of (noun_idx, particle_idx) adjacent pairs
# ---------------------------------------------------------------------------

def build_gold_entries() -> list[tuple[list[str], list[int], list[tuple[int, int]]]]:
    """Return 150 diverse, morphologically annotated Japanese sentences."""
    # Hand-curated diverse sentences covering okurigana verbs/adjectives and noun+particle boundaries.
    raw: list[list[str]] = [
        ["新しい", "トークナイザー", "の", "設計", "を", "調べる", "。"],
        ["開発者", "が", "処理", "を", "速く", "書き換える", "。"],
        ["日本", "の", "研究者", "は", "美しい", "日本語", "を", "学ぶ", "。"],
        ["東京", "で", "開か", "れた", "会議", "から", "新しい", "計画", "が", "始まる", "。"],
        ["私", "たち", "は", "昼", "に", "温かい", "料理", "を", "食べる", "。"],
        ["高速", "な", "コンパイラ", "を", "作る", "ため", "に", "メモリ", "を", "測る", "。"],
        ["入力", "データ", "が", "正しく", "動く", "か", "確かめる", "。"],
        ["古い", "設定", "を", "切り替える", "と", "処理", "が", "安定", "する", "。"],
        ["言語", "から", "構造", "を", "読み取る", "手法", "を", "考える", "。"],
        ["京都", "の", "静かな", "道", "を", "ゆっくり", "歩く", "。"],
        ["技術者", "は", "長い", "文字列", "を", "安全", "に", "取り扱う", "。"],
        ["評価", "結果", "を", "詳しく", "見直す", "こと", "が", "大切", "だ", "。"],
        ["大阪", "の", "工場", "で", "新しい", "部品", "を", "組み立てる", "。"],
        ["朝", "の", "公園", "を", "元気に", "走る", "人", "が", "増える", "。"],
        ["この", "論文", "を", "最後", "まで", "丁寧", "に", "読む", "。"],
        ["実験", "の", "記録", "を", "ノート", "へ", "正確", "に", "書く", "。"],
        ["望遠鏡", "で", "遠い", "星", "の", "光", "を", "見る", "。"],
        ["専門家", "の", "説明", "を", "集中", "して", "聞く", "。"],
        ["会議", "で", "次の", "課題", "について", "詳しく", "話す", "。"],
        ["北海道", "の", "広い", "大地", "を", "列車", "で", "旅", "する", "。"],
        ["語彙", "の", "数", "を", "増やす", "と", "圧縮率", "が", "上がる", "。"],
        ["不要", "な", "中間", "トークン", "を", "減らす", "工夫", "を", "試みる", "。"],
        ["設計", "の", "方針", "が", "大きく", "変わる", "場面", "も", "ある", "。"],
        ["重要", "な", "ログ", "だけ", "が", "手元", "に", "残る", "。"],
        ["研究", "を", "粘り強く", "進める", "姿勢", "が", "実", "を", "結ぶ", "。"],
        ["理論", "から", "実装", "へ", "知識", "を", "導く", "。"],
        ["基盤", "技術", "が", "社会", "の", "インフラ", "を", "支える", "。"],
        ["計算", "コスト", "を", "低く", "抑える", "方法", "を", "探す", "。"],
        ["郵便", "窓口", "で", "大切な", "荷物", "を", "受け取る", "。"],
        ["複数", "の", "モデル", "を", "公平", "な", "条件", "で", "比べる", "。"],
        ["困難", "な", "問題", "でも", "あきらめず", "に", "続ける", "。"],
        ["高い", "精度", "を", "求める", "なら", "データ", "の", "質", "を", "保つ", "。"],
        ["数値", "で", "性能", "の", "差", "を", "明確", "に", "示す", "。"],
        ["グラフ", "は", "時刻", "ごと", "の", "変化", "を", "表す", "。"],
        ["作業", "環境", "を", "使いやすく", "整える", "。"],
        ["不足", "し", "た", "情報", "を", "資料", "で", "補う", "。"],
        ["危険", "な", "分岐", "を", "事前", "に", "避ける", "設計", "に", "する", "。"],
        ["大量", "の", "テキスト", "を", "高速", "に", "扱う", "仕組み", "が", "動く", "。"],
        ["複雑", "な", "パズル", "を", "順番", "に", "解く", "。"],
        ["子供", "たち", "が", "校庭", "で", "楽しく", "遊ぶ", "。"],
        ["農家", "が", "新しい", "苗", "を", "春", "に", "育てる", "。"],
        ["配達員", "が", "玄関", "まで", "箱", "を", "届ける", "。"],
        ["博物館", "は", "古い", "地図", "を", "各地", "から", "集める", "。"],
        ["読者", "が", "好み", "の", "記事", "を", "自由", "に", "選ぶ", "。"],
        ["朝", "早く", "店", "の", "扉", "を", "開く", "。"],
        ["夜", "遅く", "図書館", "の", "窓", "を", "閉じる", "。"],
        ["受付", "で", "書類", "を", "担当者", "に", "渡す", "。"],
        ["季節", "の", "花", "を", "部屋", "に", "飾る", "。"],
        ["画家", "が", "海", "の", "風景", "を", "鮮やか", "に", "描く", "。"],
        ["合唱団", "が", "ホール", "で", "美しい", "歌", "を", "歌う", "。"],
        ["登山者", "が", "朝霧", "の", "山", "へ", "登る", "。"],
        ["選手", "が", "広い", "プール", "を", "力強く", "泳ぐ", "。"],
        ["休日", "は", "自宅", "で", "ゆっくり", "休む", "。"],
        ["猫", "が", "日向", "で", "気持ちよく", "眠る", "。"],
        ["友人", "と", "昔話", "で", "明るく", "笑う", "。"],
        ["札幌", "の", "冬", "は", "雪", "が", "深い", "。"],
        ["仙台", "の", "並木道", "は", "夏", "に", "涼しい", "。"],
        ["名古屋", "から", "新幹線", "で", "東京", "へ", "向かう", "。"],
        ["福岡", "の", "市場", "で", "新鮮", "な", "魚", "を", "買う", "。"],
        ["横浜", "の", "港", "に", "大きい", "船", "が", "着く", "。"],
        ["神戸", "の", "坂道", "は", "海", "の", "眺め", "が", "良い", "。"],
        ["広島", "の", "川", "は", "水", "が", "澄んで", "いる", "。"],
        ["金沢", "の", "庭園", "は", "四季", "ごと", "に", "美しい", "。"],
        ["長野", "の", "高原", "で", "美味しい", "果物", "を", "食べる", "。"],
        ["筑波", "の", "研究所", "で", "新しい", "材料", "を", "作る", "。"],
        ["アルゴリズム", "の", "改善", "が", "システム", "の", "速度", "を", "上げる", "。"],
        ["データベース", "から", "必要な", "レコード", "を", "素早く", "探す", "。"],
        ["サーバー", "の", "負荷", "が", "急", "に", "高まる", "原因", "を", "調べる", "。"],
        ["ネットワーク", "の", "遅延", "を", "最小限", "に", "抑える", "。"],
        ["ベンチマーク", "の", "数値", "を", "公平", "に", "比べる", "。"],
        ["インターフェース", "を", "分かりやすく", "書き換える", "。"],
        ["パラメータ", "の", "調整", "で", "性能", "が", "大きく", "変わる", "。"],
        ["フレームワーク", "の", "更新", "に", "合わせ", "て", "手順", "を", "見直す", "。"],
        ["エンコーダ", "は", "文字列", "を", "整数", "列", "へ", "変換", "する", "。"],
        ["デコーダ", "が", "元の", "文章", "を", "完全", "に", "復元", "する", "。"],
        ["バイナリ", "形式", "で", "モデル", "を", "安全", "に", "保存", "する", "。"],
        ["キャッシュ", "の", "効率", "が", "全体", "の", "速度", "を", "左右", "する", "。"],
        ["スクリプト", "が", "毎日", "の", "集計", "を", "自動", "で", "動かす", "。"],
        ["コーパス", "の", "偏り", "を", "事前", "に", "確かめる", "。"],
        ["言語", "モデル", "の", "学習", "に", "長い", "時間", "が", "かかる", "。"],
        ["日本語", "処理", "で", "は", "文字", "境界", "の", "扱い", "が", "難しい", "。"],
        ["開発", "チーム", "は", "バグ", "の", "原因", "を", "素早く", "見つける", "。"],
        ["利用者", "から", "届いた", "要望", "を", "一覧", "に", "まとめる", "。"],
        ["翻訳者", "が", "自然", "な", "表現", "を", "慎重", "に", "選ぶ", "。"],
        ["図書館", "の", "棚", "から", "古い", "辞典", "を", "取り出す", "。"],
        ["科学者", "が", "仮説", "を", "実験", "で", "確かめる", "。"],
        ["設計者", "は", "単純", "な", "規則", "を", "組み合わせる", "。"],
        ["編集者", "が", "文章", "の", "誤り", "を", "赤ペン", "で", "直す", "。"],
        ["学生", "は", "毎日", "新しい", "単語", "を", "覚える", "。"],
        ["教師", "が", "黒板", "に", "詳しい", "図", "を", "描く", "。"],
        ["観測所", "の", "記録", "が", "気候", "の", "変化", "を", "示す", "。"],
        ["鉄道", "会社", "が", "新しい", "車両", "を", "路線", "に", "入れる", "。"],
        ["職人", "が", "薄い", "木板", "を", "正確", "に", "削る", "。"],
        ["音楽家", "が", "静かな", "夜", "に", "ピアノ", "を", "弾く", "。"],
        ["料理人", "が", "旬", "の", "食材", "で", "美味しい", "皿", "を", "作る", "。"],
        ["建築家", "は", "強い", "耐震", "構造", "を", "考える", "。"],
        ["航海士", "が", "星", "の", "位置", "から", "進路", "を", "測る", "。"],
        ["写真家", "が", "朝日", "の", "海", "を", "カメラ", "に", "収める", "。"],
        ["記者", "が", "現場", "の", "声", "を", "記事", "に", "書く", "。"],
        ["医師", "が", "患者", "の", "体調", "を", "丁寧", "に", "調べる", "。"],
        ["店員", "が", "棚", "の", "商品", "を", "きれい", "に", "並べる", "。"],
        ["司書", "が", "返却", "さ", "れた", "本", "を", "書庫", "へ", "戻す", "。"],
        ["作業員", "が", "安全", "装置", "の", "点検", "を", "始める", "。"],
        ["運転手", "が", "交差点", "の", "手前", "で", "速度", "を", "落とす", "。"],
        ["研究", "室", "の", "照明", "を", "夜", "に", "消す", "。"],
        ["端末", "の", "画面", "に", "詳しい", "エラー", "が", "出る", "。"],
        ["テスト", "項目", "を", "一つ", "ずつ", "確実", "に", "通す", "。"],
        ["モジュール", "間", "の", "依存", "を", "できる", "だけ", "減らす", "。"],
        ["文字", "コード", "の", "変換", "で", "文字化け", "を", "防ぐ", "。"],
        ["並列", "処理", "により", "解析", "時間", "が", "短く", "なる", "。"],
        ["メモリ", "使用量", "を", "常に", "低い", "水準", "に", "保つ", "。"],
        ["ファイル", "の", "末尾", "まで", "順序", "よく", "読み込む", "。"],
        ["設定", "値", "を", "環境", "に", "合わせ", "て", "切り替える", "。"],
        ["構造体", "の", "定義", "を", "簡潔", "に", "書き換える", "。"],
        ["新しい", "アルゴリズム", "は", "従来", "より", "速く", "動く", "。"],
        ["古い", "コード", "を", "段階", "的", "に", "置き換える", "。"],
        ["正確", "な", "計測", "の", "ため", "に", "条件", "を", "そろえる", "。"],
        ["公開", "さ", "れた", "データ", "を", "誰でも", "自由", "に", "使える", "。"],
        ["仕様書", "の", "曖昧", "な", "箇所", "を", "早め", "に", "直す", "。"],
        ["開発者", "の", "負担", "を", "軽く", "する", "道具", "を", "作る", "。"],
        ["自動", "テスト", "が", "変更", "の", "安全", "を", "守る", "。"],
        ["日本", "の", "季節", "は", "四季", "の", "移ろい", "が", "美しい", "。"],
        ["京都", "の", "寺院", "で", "古い", "庭園", "を", "見る", "。"],
        ["大阪", "の", "商店街", "で", "温かい", "たこ焼き", "を", "食べる", "。"],
        ["東京", "の", "駅", "から", "空港", "まで", "電車", "で", "移動", "する", "。"],
        ["沖縄", "の", "海", "は", "夏", "も", "冬", "も", "青い", "。"],
        ["金沢", "の", "美術館", "で", "現代", "作品", "を", "楽しむ", "。"],
        ["長野", "の", "温泉", "で", "旅", "の", "疲れ", "を", "癒やす", "。"],
        ["札幌", "の", "時計台", "は", "街", "の", "象徴", "と", "して", "親しまれる", "。"],
        ["夕方", "の", "空", "が", "鮮やかな", "茜色", "に", "染まる", "。"],
        ["雨", "の", "あと", "に", "大きな", "虹", "が", "架かる", "。"],
        ["森", "の", "奥", "から", "清らかな", "小川", "が", "流れる", "。"],
        ["風", "が", "黄金色", "の", "稲穂", "を", "優しく", "揺らす", "。"],
        ["高い", "塔", "の", "上", "から", "街", "全体", "を", "見渡す", "。"],
        ["港", "の", "灯台", "が", "暗い", "海", "を", "明るく", "照らす", "。"],
        ["市場", "の", "朝", "は", "活気", "が", "あふれて", "いる", "。"],
        ["職人", "の", "手仕事", "が", "美しい", "器", "を", "生み出す", "。"],
        ["統計", "資料", "を", "地域", "ごと", "に", "集計", "する", "。"],
        ["議論", "の", "要点", "を", "短い", "文", "に", "まとめる", "。"],
        ["将来", "の", "課題", "に", "備え", "て", "計画", "を", "立てる", "。"],
        ["経験", "の", "浅い", "担当", "でも", "迷わ", "ず", "操作", "できる", "。"],
        ["細部", "まで", "注意", "を", "払い", "品質", "を", "高める", "。"],
        ["基本", "の", "手順", "を", "何度", "も", "繰り返し", "練習", "する", "。"],
        ["自然", "と", "調和", "し", "た", "暮らし", "を", "求める", "。"],
        ["多様", "な", "文化", "を", "互いに", "認め", "合い", "尊重", "する", "。"],
        ["情報", "の", "正確さ", "を", "複数", "の", "視点", "から", "確かめる", "。"],
        ["複雑", "な", "現象", "を", "単純", "な", "法則", "で", "説明する", "。"],
        ["限られた", "資源", "を", "大切", "に", "使い", "未来", "へ", "残す", "。"],
        ["技術", "の", "進歩", "が", "新しい", "産業", "を", "切り開く", "。"],
        ["今日", "の", "学び", "を", "明日", "の", "仕事", "に", "活かす", "。"],
    ]

    # Identify okurigana words and noun+particle pairs automatically and deterministically
    particles = {"は", "が", "を", "に", "の", "で", "と", "へ", "から", "まで", "より", "も"}

    def is_kanji_char(ch: str) -> bool:
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x2FA1F
            or 0x3005 <= cp <= 0x3007
        )

    def is_hira_char(ch: str) -> bool:
        cp = ord(ch)
        return (0x3041 <= cp <= 0x3096) or (0x309D <= cp <= 0x309F)

    def is_kata_char(ch: str) -> bool:
        cp = ord(ch)
        return (0x30A1 <= cp <= 0x30FA) or (0x30FD <= cp <= 0x30FF) or cp == 0x30FC

    def is_okurigana_word(tok: str) -> bool:
        if len(tok) < 2:
            return False
        return is_kanji_char(tok[0]) and is_hira_char(tok[-1]) and any(is_hira_char(c) for c in tok[1:])

    def is_noun_like(tok: str) -> bool:
        if not tok or tok in particles or tok in {"。", "、", "？", "！"}:
            return False
        # Kanji noun, Katakana noun, or mixed Kanji+Katakana noun (no trailing okurigana unless known)
        return all(is_kanji_char(c) or is_kata_char(c) for c in tok)

    entries: list[tuple[list[str], list[int], list[tuple[int, int]]]] = []
    for morphs in raw:
        oku_idxs = [i for i, m in enumerate(morphs) if is_okurigana_word(m)]
        np_pairs = [
            (i, i + 1)
            for i in range(len(morphs) - 1)
            if is_noun_like(morphs[i]) and morphs[i + 1] in particles
        ]
        entries.append((morphs, oku_idxs, np_pairs))
    return entries


# ---------------------------------------------------------------------------
# Generators for unique, diverse, non-repeating corpora
# ---------------------------------------------------------------------------

def generate_ja_pool(rng: random.Random, target_count: int) -> list[str]:
    """Generate `target_count` unique, natural Japanese sentences across 4 genres."""
    seen: set[str] = set()
    out: list[str] = []

    years = [str(y) for y in range(1998, 2027)]
    months = [f"{m}月" for m in range(1, 13)]
    counts = ["3件", "8回", "12種", "24時間", "40項目", "64文字", "128個", "256バイト", "4096語"]
    modifiers = [
        "実験環境において", "実務の現場では", "長期間の評価によると", "前回の調査と比べて",
        "分散配置のもとで", "最新の仕様に基づき", "標準的な手順に従い", "限定された条件下でも",
        "高負荷な状況下で", "複数の言語が混ざる場面で", "季節の変わり目には", "静かな環境の中で",
    ]
    connectors = [
        "その結果、", "一方で、", "さらに、", "また、", "したがって、", "同時に、",
        "なお、", "具体的には、", "実際に、", "これに対して、",
    ]

    attempts = 0
    max_attempts = target_count * 20
    while len(out) < target_count and attempts < max_attempts:
        attempts += 1
        genre = attempts % 4
        if genre == 0:
            # Technical explanation
            subj = rng.choice(JA_TECH_SUBJECTS)
            obj = rng.choice(JA_TECH_OBJECTS)
            adj_p, adj_adv = rng.choice(JA_ADJECTIVES)
            v_dict, v_past, v_te, v_prog = rng.choice(JA_VERBS_OKURIGANA)
            role = rng.choice(JA_ROLES)
            mod = rng.choice(modifiers)
            cnt = rng.choice(counts)
            pat = attempts % 5
            if pat == 0:
                s = f"{mod}{role}は{adj_p}{subj}を用いて{obj}を{adj_adv}{v_dict}。"
            elif pat == 1:
                s = f"{subj}の{obj}を{cnt}単位で{v_te}から、全体の振る舞いを{v_past}。"
            elif pat == 2:
                s = f"「{subj}」に関する検証では、{obj}の扱いが{adj_adv}なる点を{v_prog}。"
            elif pat == 3:
                subj2 = rng.choice(JA_TECH_SUBJECTS)
                s = f"{subj}と{subj2}を比べた結果、{obj}の負荷を{adj_adv}抑えて{v_dict}ことが分かった。"
            else:
                s = f"{rng.choice(connectors)}{adj_p}{obj}を{v_dict}ため、{subj}の内部状態を{v_prog}。"
        elif genre == 1:
            # Wikipedia-style encyclopedia
            topic, detail, org, activity = rng.choice(JA_ENCYCLOPEDIA_TOPICS)
            place = rng.choice(JA_PLACES)
            yr = rng.choice(years)
            mo = rng.choice(months)
            adj_p, adj_adv = rng.choice(JA_ADJECTIVES)
            v_dict, v_past, v_te, v_prog = rng.choice(JA_VERBS_OKURIGANA)
            pat = attempts % 4
            if pat == 0:
                s = f"{topic}（{place}）は{yr}年{mo}に{org}が{v_past}{adj_p}仕組みであり、{detail}を{v_dict}。"
            elif pat == 1:
                s = f"{yr}年以降、{place}の{org}は{topic}における{detail}の{activity}を{adj_adv}{v_prog}。"
            elif pat == 2:
                role = rng.choice(JA_ROLES)
                s = f"{topic}の研究において{role}は{detail}を重視し、{activity}の記録を{v_te}{v_dict}。"
            else:
                s = f"『{topic}年鑑』によると、{place}では{detail}に関する{activity}が{yr}年から{v_prog}。"
        elif genre == 2:
            # News / editorial prose
            place = rng.choice(JA_PLACES)
            role = rng.choice(JA_ROLES)
            subj = rng.choice(JA_TECH_SUBJECTS)
            v_dict, v_past, v_te, v_prog = rng.choice(JA_VERBS_OKURIGANA)
            adj_p, adj_adv = rng.choice(JA_ADJECTIVES)
            mo = rng.choice(months)
            pat = attempts % 4
            if pat == 0:
                s = f"{place}の{role}グループは{mo}、{adj_p}{subj}の導入に向けて現場の課題を{v_past}。"
            elif pat == 1:
                s = f"{rng.choice(connectors)}{place}周辺では{subj}への関心が高まり、{role}が体制を{adj_adv}{v_prog}。"
            elif pat == 2:
                s = f"報告書は「{adj_p}基準を{v_dict}必要がある」と指摘し、{mo}末までに方針を{v_dict}と述べた。"
            else:
                cnt = rng.choice(counts)
                s = f"調査では{cnt}の対象について比較を行い、{place}における改善幅を{v_te}{v_past}。"
        else:
            # Literary / everyday prose
            scene = rng.choice(JA_LITERARY_NOUNS)
            food = rng.choice(JA_FOODS)
            place = rng.choice(JA_PLACES)
            adj_p, adj_adv = rng.choice(JA_ADJECTIVES)
            dv_dict, dv_past, dv_te, dv_prog = rng.choice(JA_DAILY_VERBS)
            v_dict, v_past, v_te, v_prog = rng.choice(JA_VERBS_OKURIGANA)
            pat = attempts % 4
            if pat == 0:
                s = f"{place}の{scene}を眺めながら、旅人は{food}を静かに{dv_past}。"
            elif pat == 1:
                s = f"{adj_p}空の下で、人々はその日の出来事を思い出しつつ{scene}のそばを{dv_prog}。"
            elif pat == 2:
                s = f"季節が巡るたびに{scene}の色合いは変わり、訪れる人の心に{adj_p}印象を{v_dict}。"
            else:
                s = f"長い一日を終えたあと、仲間たちは{food}を囲んで{adj_adv}笑い合い、明日への話を{v_past}。"

        if s not in seen:
            seen.add(s)
            out.append(s)

    return out


def generate_en_pool(rng: random.Random, target_count: int) -> list[str]:
    """Generate `target_count` unique English technical/systems/ML sentences."""
    seen: set[str] = set()
    out: list[str] = []
    versions = ["v1.4", "v2.0", "v3.1", "4096-vocab", "8192-vocab", "single-thread", "UTF-8", "phase-2"]
    attempts = 0
    while len(out) < target_count and attempts < target_count * 20:
        attempts += 1
        subj = rng.choice(EN_SUBJECTS)
        verb = rng.choice(EN_VERBS)
        obj = rng.choice(EN_OBJECTS)
        clause = rng.choice(EN_CLAUSES)
        adv = rng.choice(EN_ADVERBS)
        ver = rng.choice(versions)
        pat = attempts % 4
        if pat == 0:
            s = f"{subj} {verb} {obj} {clause}."
        elif pat == 1:
            s = f"{adv} {subj.lower()} ({ver}) {verb} {obj} {clause}."
        elif pat == 2:
            subj2 = rng.choice(EN_SUBJECTS).lower()
            s = f"When paired with {subj2}, {subj.lower()} {verb} {obj} {clause}."
        else:
            s = f"In the {ver} configuration, {subj.lower()} {verb} {obj} {clause}."
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def generate_code_pool(rng: random.Random, target_count: int) -> list[str]:
    """Generate `target_count` unique realistic Rust & Python code snippets."""
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < target_count and attempts < target_count * 20:
        attempts += 1
        if attempts % 2 == 0:
            ty = rng.choice(RUST_TYPES)
            f1, f2 = rng.sample(RUST_FIELDS, 2)
            fn = rng.choice(RUST_FUNCS)
            limit = rng.choice([32, 64, 128, 256, 512, 1024, 4096])
            idx = attempts
            snippet = (
                f"/// Deterministic worker {idx} for `{ty}`.\n"
                f"pub struct {ty}{idx}<'a> {{\n"
                f"    pub {f1}: &'a [u8],\n"
                f"    pub {f2}: usize,\n"
                f"}}\n\n"
                f"impl<'a> {ty}{idx}<'a> {{\n"
                f"    pub fn {fn}(&self, cap: usize) -> Result<Vec<u32>, &'static str> {{\n"
                f"        if self.{f1}.len() > cap.max({limit}) && self.{f2} == 0 {{\n"
                f"            return Err(\"capacity exceeded\");\n"
                f"        }}\n"
                f"        Ok(self.{f1}.iter().take({limit}).map(|&b| b as u32).collect())\n"
                f"    }}\n"
                f"}}\n"
            )
        else:
            cls = rng.choice(PY_CLASSES)
            fn = rng.choice(PY_FUNCS)
            f1, f2 = rng.sample(RUST_FIELDS, 2)
            thr = rng.choice([0.05, 0.1, 0.25, 0.5, 0.85, 0.95])
            idx = attempts
            snippet = (
                f"@dataclass(frozen=True)\n"
                f"class {cls}{idx}:\n"
                f"    \"\"\"Immutable evaluation record #{idx}.\"\"\"\n"
                f"    {f1}: int\n"
                f"    {f2}: float\n"
                f"    tags: tuple[str, ...]\n\n"
                f"def {fn}_{idx}(items: list[{cls}{idx}], min_score: float = {thr}) -> dict[str, float]:\n"
                f"    filtered = [x for x in items if x.{f2} >= min_score and x.{f1} > 0]\n"
                f"    return {{t: round(sum(x.{f2} for x in filtered), 4) for x in filtered for t in x.tags}}\n"
            )
        if snippet not in seen:
            seen.add(snippet)
            out.append(snippet)
    return out


def generate_mixed_pool(rng: random.Random, target_count: int) -> list[str]:
    """Generate `target_count` unique Markdown+JP+Code documentation blocks."""
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < target_count and attempts < target_count * 20:
        attempts += 1
        subj = rng.choice(JA_TECH_SUBJECTS)
        obj = rng.choice(JA_TECH_OBJECTS)
        v_dict, _, _, _ = rng.choice(JA_VERBS_OKURIGANA)
        fn = rng.choice(RUST_FUNCS)
        ty = rng.choice(RUST_TYPES)
        limit = rng.choice([64, 128, 256, 512])
        idx = attempts
        block = (
            f"### {idx}. {subj} と `{fn}` の連携\n\n"
            f"- 対象: `{ty}` を経由して{obj}を{v_dict}。\n"
            f"- 上限: チャンク長は最大 {limit} 文字に制限する。\n\n"
            f"```rust\n"
            f"// {subj}の境界検査 ({idx})\n"
            f"let mut items_{idx}: Vec<u32> = Vec::with_capacity({limit});\n"
            f"for (pos, &byte) in input.iter().enumerate() {{\n"
            f"    if pos < {limit} && byte != 0 {{ items_{idx}.push(byte as u32); }}\n"
            f"}}\n"
            f"```\n"
        )
        if block not in seen:
            seen.add(block)
            out.append(block)
    return out


def join_to_target_bytes(items: list[str], target_bytes: int, sep: str = "\n") -> tuple[list[str], list[str]]:
    """Take prefix of `items` until >= `target_bytes`, returning (taken, remaining)."""
    taken: list[str] = []
    total = 0
    sep_len = len(sep.encode("utf-8"))
    for idx, item in enumerate(items):
        b = len(item.encode("utf-8")) + sep_len
        taken.append(item)
        total += b
        if total >= target_bytes:
            return taken, items[idx + 1 :]
    return taken, []


def verify_with_sudachipy(gold_rows: list[dict]) -> dict[str, int]:
    """Optional diagnostic cross-check with sudachipy if installed."""
    try:
        from sudachipy import dictionary, tokenizer  # type: ignore
    except Exception:
        return {"available": 0}

    tok = dictionary.Dictionary().tokenizer()
    mode = tokenizer.Tokenizer.SplitMode.C
    exact_matches = 0
    for row in gold_rows:
        sudachi_surfaces = [m.surface() for m in tok.tokenize(row["text"], mode)]
        if sudachi_surfaces == row["tokens"]:
            exact_matches += 1
    return {"available": 1, "checked": len(gold_rows), "sudachi_exact_matches": exact_matches}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261122)

    # 1. Build gold boundary dataset first so we can exclude any overlap
    gold_raw = build_gold_entries()
    gold_texts = {"".join(m) for m, _, _ in gold_raw}

    # 2. Generate large unique domain pools
    ja_pool = [s for s in generate_ja_pool(rng, int(16500 * TRAIN_SCALE)) if s not in gold_texts]
    en_pool = generate_en_pool(rng, int(7000 * TRAIN_SCALE))
    code_pool = generate_code_pool(rng, int(2200 * TRAIN_SCALE))
    mixed_pool = generate_mixed_pool(rng, int(1600 * TRAIN_SCALE))

    rng.shuffle(ja_pool)
    rng.shuffle(en_pool)
    rng.shuffle(code_pool)
    rng.shuffle(mixed_pool)

    # 3. Partition strictly disjoint train vs held-out test splits
    # Target train ~3.25 MB x TRAIN_SCALE: JA 55%, EN 20%, Code 15%, Mixed 10%
    train_ja, rem_ja = join_to_target_bytes(ja_pool, int(1_780_000 * TRAIN_SCALE))
    train_en, rem_en = join_to_target_bytes(en_pool, int(650_000 * TRAIN_SCALE))
    train_code, rem_code = join_to_target_bytes(code_pool, int(490_000 * TRAIN_SCALE))
    train_mixed, rem_mixed = join_to_target_bytes(mixed_pool, int(330_000 * TRAIN_SCALE))

    # Held-out test splits
    test_ja, _ = join_to_target_bytes(rem_ja, 200_000)
    test_en, _ = join_to_target_bytes(rem_en, 100_000)
    test_code, _ = join_to_target_bytes(rem_code, 100_000)
    test_mixed, _ = join_to_target_bytes(rem_mixed, 100_000)

    # Disjointness assertions
    assert set(train_ja).isdisjoint(test_ja)
    assert set(train_en).isdisjoint(test_en)
    assert set(train_code).isdisjoint(test_code)
    assert set(train_mixed).isdisjoint(test_mixed)
    assert set(train_ja).isdisjoint(gold_texts)
    assert set(test_ja).isdisjoint(gold_texts)

    # Interleave train splits deterministically by paragraph blocks
    train_blocks: list[str] = []
    for i in range(0, len(train_ja), 8):
        train_blocks.append("\n".join(train_ja[i : i + 8]))
    for i in range(0, len(train_en), 6):
        train_blocks.append("\n".join(train_en[i : i + 6]))
    for item in train_code:
        train_blocks.append(item.rstrip("\n"))
    for item in train_mixed:
        train_blocks.append(item.rstrip("\n"))
    rng.shuffle(train_blocks)
    train_text = "\n\n".join(train_blocks) + "\n"

    (DATA_DIR / "train_corpus.txt").write_text(train_text, encoding="utf-8")
    (DATA_DIR / "test_ja_prose.txt").write_text("\n".join(test_ja) + "\n", encoding="utf-8")
    (DATA_DIR / "test_en_prose.txt").write_text("\n".join(test_en) + "\n", encoding="utf-8")
    (DATA_DIR / "test_code.txt").write_text("\n".join(test_code) + "\n", encoding="utf-8")
    (DATA_DIR / "test_mixed_ja_code.txt").write_text("\n".join(test_mixed) + "\n", encoding="utf-8")

    # 4. Adversarial 1 MiB input: 960,000 bytes 'あ' + 40,000 bytes '🦀' + 48,576 bytes 'a' = 1,048,576 bytes
    adv_text = ("あ" * 320_000) + ("🦀" * 10_000) + ("a" * 48_576)
    assert len(adv_text.encode("utf-8")) == 1_048_576
    (DATA_DIR / "adversarial_1mb.txt").write_text(adv_text, encoding="utf-8")

    # 5. Write ja_boundary_gold.jsonl
    gold_rows: list[dict] = []
    for idx, (morphs, oku_idxs, np_pairs) in enumerate(gold_raw):
        text = "".join(morphs)
        char_spans: list[list[int]] = []
        byte_spans: list[list[int]] = []
        c_pos = 0
        b_pos = 0
        for m in morphs:
            c_len = len(m)
            b_len = len(m.encode("utf-8"))
            char_spans.append([c_pos, c_pos + c_len])
            byte_spans.append([b_pos, b_pos + b_len])
            c_pos += c_len
            b_pos += b_len

        okurigana_words = [
            {
                "text": morphs[i],
                "char_span": char_spans[i],
                "byte_span": byte_spans[i],
            }
            for i in oku_idxs
        ]
        noun_particles = [
            {
                "noun": morphs[ni],
                "particle": morphs[pi],
                "boundary_char": char_spans[ni][1],
                "boundary_byte": byte_spans[ni][1],
                "combined_char_span": [char_spans[ni][0], char_spans[pi][1]],
                "combined_byte_span": [byte_spans[ni][0], byte_spans[pi][1]],
            }
            for ni, pi in np_pairs
        ]

        gold_rows.append(
            {
                "id": idx,
                "text": text,
                "tokens": morphs,
                "spans": char_spans,
                "byte_spans": byte_spans,
                "okurigana_words": okurigana_words,
                "noun_particles": noun_particles,
            }
        )

    gold_path = DATA_DIR / "ja_boundary_gold.jsonl"
    gold_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gold_rows) + "\n",
        encoding="utf-8",
    )

    sudachi_info = verify_with_sudachipy(gold_rows)

    # Print manifest summary
    manifest = {
        "data_dir": str(DATA_DIR),
        "files": {},
        "gold_sentences": len(gold_rows),
        "gold_okurigana_words": sum(len(r["okurigana_words"]) for r in gold_rows),
        "gold_noun_particles": sum(len(r["noun_particles"]) for r in gold_rows),
        "sudachipy_check": sudachi_info,
    }
    for fname in [
        "train_corpus.txt",
        "test_ja_prose.txt",
        "test_en_prose.txt",
        "test_code.txt",
        "test_mixed_ja_code.txt",
        "adversarial_1mb.txt",
        "ja_boundary_gold.jsonl",
    ]:
        p = DATA_DIR / fname
        raw_bytes = p.read_bytes()
        manifest["files"][fname] = {
            "bytes": len(raw_bytes),
            "sha256_prefix": hashlib.sha256(raw_bytes).hexdigest()[:12],
        }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
