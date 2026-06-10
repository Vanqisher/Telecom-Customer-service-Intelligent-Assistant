# =============================================================================
# WE Intelligent Assistant — RAG Pipeline
# =============================================================================
# النظام ده عبارة عن chatbot ذكي لشركة WE (المصرية للاتصالات).
# بيشتغل بنظام RAG (Retrieval-Augmented Generation):
#   1. بيجيب أكتر الـ chunks صلة من قاعدة البيانات
#   2. بيديها للـ LLM كـ context عشان يجاوب بدقة
#
# المكونات الرئيسية:
#   - Config          : كل الإعدادات في مكان واحد
#   - Normalization   : تنظيف النص العربي للـ BM25
#   - Query Router    : يوجّه السؤال للـ domain الصح
#   - FlagReranker    : يرتب النتائج بعد الاسترجاع
#   - RAGPipeline     : يجمع Chroma + BM25 + Reranker
#   - WEAssistant     : الواجهة اللي بتتكلم مع العميل
# =============================================================================

import json
import os
import re
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
import warnings
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

# -----------------------------------------------------------------------------
# محاولة تحميل مكتبة الـ Reranker (اختيارية)
# لو مش موجودة، النظام هيشتغل عادي بدونها وهيرجع للترتيب الافتراضي (RRF)
# -----------------------------------------------------------------------------
try:
    from FlagEmbedding import FlagReranker as _FlagReranker

    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

# -----------------------------------------------------------------------------
# إعدادات عامة للبيئة
# بنوقف warnings مزعجة، وبنوقف الـ parallelism في الـ tokenizer
# عشان ما يعملش تعارض في الـ threads، وبنحمل متغيرات الـ .env
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# -----------------------------------------------------------------------------
# إعداد الـ Logger
# بيكتب كل event بالوقت والمستوى (INFO / WARNING / ERROR)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("WE-RAG")


# =============================================================================
# CONFIG — كل إعدادات النظام في dataclass واحدة
# =============================================================================
# بدل ما الـ hardcoded values تكون مبعثرة في الكود،
# كلها مجمعة هنا وبتتقرأ من الـ .env لو موجودة.
#
# أهم الإعدادات:
#   - embed_model     : موديل الـ embeddings (BGE-M3 متعدد اللغات)
#   - rerank_model    : موديل الـ reranking لترتيب النتائج
#   - top_k_retrieve  : عدد الـ chunks اللي بنجيبها أول ما نبحث (30)
#   - top_k_final     : عدد الـ chunks اللي بنبعتها للـ LLM بعد الـ reranking (15)
#   - bm25_weight     : وزن الـ BM25 في الـ Ensemble (0.6 — أثقل عشان النص عربي)
#   - dense_weight    : وزن الـ Dense Vector Search (0.4)
# =============================================================================
@dataclass
class Config:
    openrouter_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_KEY", ""))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db"))
    data_file: str = field(
        default_factory=lambda: os.getenv("DATA_FILE",
                                          "./ProcessedData/processed_all_no_branches.json"))  # [FIX-1] updated filename

    collection_name: str = "WE_Intelligent_Assistant_Enhanced"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "openai/gpt-4o-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"

    top_k_retrieve: int = 30
    top_k_final: int = 15
    batch_size: int = 32
    history_condense: int = 6
    history_llm: int = 10

    bm25_weight: float = 0.6
    dense_weight: float = 0.4


cfg = Config()

# =============================================================================
# ARABIC NORMALIZATION & TOKENIZATION
# =============================================================================
# الـ BM25 بيشتغل على مستوى الكلمات، فلازم نوحد شكل الكلمات العربية قبل ما
# يبني الـ index أو يبحث — لأن "أحمد" و"احمد" و"إحمد" نفس الكلمة.
#
# _ARABIC_NORMALIZE: قائمة من الـ replacements:
#   - كل أشكال الألف (أ إ آ ا) → ا
#   - التاء المربوطة (ة) → هاء عادية (ه)
#   - الألف المقصورة (ى) → ياء (ي)
#   - المسافات المتعددة → مسافة واحدة
# =============================================================================
_ARABIC_NORMALIZE = [
    (r"[أإآا]", "ا"),
    (r"ة", "ه"),
    (r"ى", "ي"),
    (r"\s+", " "),
]


def normalize_arabic(text: str) -> str:
    for pattern, repl in _ARABIC_NORMALIZE:
        text = re.sub(pattern, repl, text)
    return text.strip()


# -----------------------------------------------------------------------------
# tokenize()
# الدالة دي بتحوّل أي نص لـ list من الكلمات النظيفة للـ BM25.
# بتعمل 3 حاجات:
#   1. بتشيل الـ labels الهيكلية (العنوان: / السعر: / Title: إلخ)
#      عشان الـ BM25 يـ match على المحتوى الفعلي مش على العناوين
#   2. بتعمل Arabic normalization عبر normalize_arabic()
#   3. بتستخرج الكلمات العربية والإنجليزية والأرقام بـ regex
# -----------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    # [FIX-2] Added Arabic labels to the strip pattern since text labels are now Arabic
    text = re.sub(
        r"^(Title|Tagline|Question|Answer|Code|Cost|Service Name|Category|"
        r"Step \d+|Term \d+|Warning \d+|Hotline \d+|Support Email \d+|"
        r"Permissions|Network Condition|Working Hours|Start Time|End Time|"
        r"Program Overview|Description|Items|"
        # Arabic labels added after JSON fix
        r"العنوان|السعر|المميزات|التفاصيل|الملخص|الوصف|العناصر|"
        r"العنوان الفرعي|طريقة الاشتراك|المميزات المجانية|المميزات المدفوعة|"
        r"وقت التفعيل|الرسوم الشهرية|الرسوم الربع سنوية|طريقة الفوترة|"
        r"الأذونات المطلوبة|متطلبات الشبكة|مواعيد العمل|"
        r"اسم الخدمة|الكود|التكلفة|إنجاز رئيسي|"
        r"مستوى النقاط|معدل التجميع|قاعدة النقاط|"
        r"تحذير \d+|شرط \d+|خطوة \d+|"
        r"المتطلبات|كود التفعيل|قواعد الاستخدام|"
        r"الهدف|الأهمية التاريخية|الموقع|الأقسام|"
        r"عنوان العقد|الأطراف|مدة العقد|الإجراء|العقوبة):\s*",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = normalize_arabic(text.lower())
    return re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", text)


# -----------------------------------------------------------------------------
# is_arabic()
# بتكشف إذا كان النص يحتوي على حروف عربية.
# بتُستخدم في WEAssistant عشان تحدد لغة العميل وتضبط الرد بناءً عليها.
# -----------------------------------------------------------------------------
def is_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))


# =============================================================================
# AMBIGUITY DETECTION & INTENT STRIPPER
# =============================================================================
# قبل ما نبعت السؤال للـ retriever، بنعمل عليه خطوتين:
#
# 1. AMBIGUITY DETECTION — is_ambiguous()
#    بتكشف لو السؤال غامض ومحتاج context من التاريخ عشان يتفهم.
#    السؤال غامض لو:
#      - قصير جداً (أقل من 20 حرف)
#      - بيبدأ بحروف ربط (و / ف / لكن / طب / كمان...)
#      - فيه ضمائر مبهمة (ده / دي / هو / هي / بتاعه...)
#
# 2. INTENT STRIPPER — strip_intent()
#    بتشيل الجملة التمهيدية من السؤال وتسيب الجوهر بس.
#    مثال: "عايز أعرف عن باقات الإنترنت" → "باقات الإنترنت"
#    ده بيخلي الـ retriever يبحث على المحتوى الفعلي مش على قصد العميل.
# =============================================================================
_AMBIGUITY_STARTERS = re.compile(
    r"^(و|ف|لكن|طب|طيب|وإيه|وهو|وهي|وهم|كمان|بردو|والـ|وال)",
    re.UNICODE,
)
_AMBIGUOUS_PRONOUNS = re.compile(
    r"\b(ده|دي|دول|هو|هي|هم|هن|بتاعه|بتاعها|بتاعهم|فيه|عليه|عليها|منه|منها)\b",
    re.UNICODE,
)


def is_ambiguous(query: str) -> bool:
    q = query.strip()
    return (
            len(q) < 20
            or bool(_AMBIGUITY_STARTERS.match(q))
            or bool(_AMBIGUOUS_PRONOUNS.search(q))
    )


_INTENT_PATTERNS = [
    r"^كنت (عايز|عاوز|أريد|اريد|بدي)\s+(اسأل|أسأل|أعرف|اعرف|أسال)\s+(عن|في|على|ل)\s+",
    r"^(عايز|عاوز|أريد|اريد|بدي)\s+(اسأل|أسأل|أعرف|اعرف)\s+(عن|في|على|ل)\s+",
    r"^(عايز|عاوز|أريد|اريد)\s+(أعرف|اعرف|أفهم|افهم)\s+(عن|في|على|ل)?\s*",
    r"^(ممكن|تقدر|تقدري)\s+(تقولي|تعرفني|تشرحلي|تفهمني)\s+(عن|في|على|ل)?\s*",
    r"^(محتاج|محتاجة|أحتاج)\s+(أعرف|اعرف|أفهم|معلومات)\s+(عن|في|على|ل)?\s*",
    r"^(سؤال|عندي سؤال|لو سمحت)\s*(عن|في|ل)?\s*",
    r"^(اسألك|بسألك|هسألك)\s+(عن|في)?\s*",
    r"^(قولي|قولي بقي)\s+(عن|في|ل|على)?\s*",
    r"^(حابب|حاببة)\s+(أعرف|اعرف)?\s*(عن|في|على)?\s*",
    r"^i\s+(want|wanted|need|would like)\s+to\s+(know|ask|understand|learn)\s+(about|regarding)?\s*",
    r"^can\s+you\s+(tell|explain|help)\s+(me)?\s*(about|regarding)?\s*",
    r"^(tell|explain|help)\s+(me)?\s*(about|regarding|with)?\s*",
]


def strip_intent(query: str) -> str:
    q = query.strip()
    for pattern in _INTENT_PATTERNS:
        cleaned = re.sub(pattern, "", q, flags=re.IGNORECASE | re.UNICODE).strip()
        if cleaned and cleaned != q and len(cleaned) >= 2:
            return cleaned
    return q


# =============================================================================
# METADATA-AWARE QUERY ROUTER
# =============================================================================
# بدل ما نبحث في كل الـ 266 document في كل سؤال،
# الـ Router بيحلل السؤال ويحدد الـ domain الأنسب فوراً.
# النتيجة: filter بيتبعت للـ Chroma والـ BM25 يضيّق نطاق البحث.
#
# كيف بيشتغل:
#   - كل domain عنده كلمات مفتاحية (signals) مجمعة في regex
#   - لو السؤال match مع signal → بيرجع filter dict
#   - الترتيب مهم: الأكثر تخصصاً (معاك / WE Air) قبل الأعم (USSD / payments)
#   - لو مفيش match → بيرجع None ويبحث في كل الـ corpus
#
# الـ Signals الموجودة:
#   _USSD_SIGNALS        : أكواد USSD والباقات
#   _FAQ_SIGNALS         : أسئلة إجرائية (كيف / ازاي / خطوات)
#   _PAYMENT_SIGNALS     : طرق الدفع والفواتير
#   _MAAK_SIGNALS        : خدمة معاك للصم وضعاف السمع
#   _BONUS_SIGNALS       : برنامج النقاط والمكافآت
#   _ENTERTAINMENT_SIGNALS: خدمات الترفيه (WatchIT / كول تون / WE Sports)
#   _WE_AIR_SIGNALS      : إنترنت WE Air الهوائي
#   _WE_AIR_PREPAID_SIGNALS: WE Air بالدفع المسبق تحديداً
#   _5G_SIGNALS          : أسئلة عن الجيل الخامس
#   _GOVERNMENT_SIGNALS  : خدمات حكومية (تموين / نقل رقم)
#   _GREETINGS_PATTERN   : تحيات (مش بيتبعت للـ router، بتتعامل معاها بشكل خاص)
# =============================================================================
_USSD_SIGNALS = re.compile(
    r"(كود|رمز|\*\d|ussd|اشتراك|تجديد|إلغاء|سلفني|كلمني|رصيد|باقة|"
    r"اشترك|تشترك|تفعيل|تفعل)",
    re.IGNORECASE | re.UNICODE,
)
_FAQ_SIGNALS = re.compile(
    r"(كيف|ازاي|إزاي|خطوات|طريقة|هل يمكن|ممكن|what|how|can i|"
    r"فاتورة|سداد|شحن|نقل ملكية|تغيير|استفسار)",
    re.IGNORECASE | re.UNICODE,
)
_PAYMENT_SIGNALS = re.compile(
    r"(دفع|سداد|فوري|مصاري|فيزا|كريدت|بطاقة|محفظة|pay|payment|فاتورة)",
    re.IGNORECASE | re.UNICODE,
)
_MAAK_SIGNALS = re.compile(
    r"(معاك|صم|ضعاف السمع|إشارة|ma3ak|sign language|deaf)",
    re.IGNORECASE | re.UNICODE,
)
_BONUS_SIGNALS = re.compile(
    r"(بونص|bonus|نقاط|مكافأة|مكافآت|points|deals)",
    re.IGNORECASE | re.UNICODE,
)
# [FIX-3] New signal patterns for domains added during JSON fix
_ENTERTAINMENT_SIGNALS = re.compile(
    r"(watch.?it|كول.?تون|العب.?واكسب|وي.?سبورتس|we.?sports|ترفيه|"
    r"افلام|مسلسلات|نغمة|رنة|العاب|contest|entertainment)",
    re.IGNORECASE | re.UNICODE,
)
_WE_AIR_SIGNALS = re.compile(
    r"(we.?air|وي.?اير|راوتر.?هوائي|انترنت.?هوائي|نت.?هوائي|"
    r"4g.?router|5g.?router|portable.?wifi|محمول.?واي.?فاي)",
    re.IGNORECASE | re.UNICODE,
)
_WE_AIR_PREPAID_SIGNALS = re.compile(
    r"(we.?air.*(مسبق|prepaid|كارت|شحن)|"
    r"(مسبق|prepaid).*(we.?air|وي.?اير))",
    re.IGNORECASE | re.UNICODE,
)
_5G_SIGNALS = re.compile(
    r"(5g|الجيل.?الخامس|خامس)",
    re.IGNORECASE | re.UNICODE,
)
_GOVERNMENT_SIGNALS = re.compile(
    r"(تموين|بطاقة.?تموين|نقل.?رقم|number.?port)",
    re.IGNORECASE | re.UNICODE,
)

_GREETINGS_PATTERN = re.compile(
    r"^(Hello|Hey|Hi|hello|hi|hey|greetings|سلام عليكم | هلا|مرحبا|مرحباً|اهلا|أهلاً|سلام|السلام عليكم|صباح الخير|مساء الخير)\s*\??$",
    re.IGNORECASE | re.UNICODE,
)


def get_metadata_filter(query: str) -> Optional[dict]:
    # [FIX-4] Extended router with new domains + sub_domain support for 5G

    if _MAAK_SIGNALS.search(query):
        return {"_source_file": "normalized_ma3ak_service.json"}

    # WE Air: check prepaid vs postpaid first for precision
    if _WE_AIR_SIGNALS.search(query):
        if _WE_AIR_PREPAID_SIGNALS.search(query):
            return {"payment_type": "prepaid"}
        return {"domain": "internet_services"}

    # 5G FAQs now have sub_domain='5G' after the JSON fix
    if _5G_SIGNALS.search(query):
        return {"sub_domain": "5G"}

    if _ENTERTAINMENT_SIGNALS.search(query):
        return {"domain": "entertainment_services"}

    if _GOVERNMENT_SIGNALS.search(query):
        return {"domain": "government_services"}

    if _USSD_SIGNALS.search(query) and not _FAQ_SIGNALS.search(query):
        return {"domain": "ussd_codes"}

    if _PAYMENT_SIGNALS.search(query) and not _FAQ_SIGNALS.search(query):
        return {"domain": "payments"}

    if _BONUS_SIGNALS.search(query):
        return {"domain": "offers_promotions"}

    return None


# =============================================================================
# FLAG RERANKER — FlagRerankerCompressor
# =============================================================================
# بعد ما الـ Ensemble Retriever يجيب أول 30 chunk،
# الـ Reranker بيرتبهم تاني بدقة أعلى ويرجع أفضل 15 بس للـ LLM.
#
# ليه محتاجينه؟
#   الـ BM25 والـ Dense Search بيعملوا matching سريع لكن مش دايماً دقيق.
#   الـ Reranker بيحسب relevance score لكل (query, chunk) pair بشكل أعمق.
#
# __init__:
#   بيحاول يحمّل الـ FlagReranker — لو فشل (مكتبة مش موجودة)
#   بيفضل يشتغل عادي بالترتيب اللي جه من RRF بدون reranking.
#
# compress_documents:
#   1. بيشيل الـ duplicates الأول (عن طريق MD5 hash للمحتوى)
#   2. بيعمل pairs (query, chunk) لكل document
#   3. بيحسب الـ scores ويرتب تنازلياً
#   4. بيرجع أفضل top_n فقط
# =============================================================================
class FlagRerankerCompressor(BaseDocumentCompressor):
    model_name: str = cfg.rerank_model
    top_n: int = cfg.top_k_final
    _reranker: Any = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if _RERANKER_AVAILABLE:
            try:
                self._reranker = _FlagReranker(self.model_name, use_fp16=False)
                log.info("FlagReranker loaded ✅")
            except Exception as e:
                log.warning(f"FlagReranker init failed: {e} — falling back to RRF order")

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Any] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        seen, unique_docs = set(), []
        for doc in documents:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

        if not self._reranker:
            return unique_docs[: self.top_n]

        pairs = [[query, doc.page_content] for doc in unique_docs]
        try:
            scores = self._reranker.compute_score(pairs)
            scored = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[: self.top_n]]
        except Exception as e:
            log.error(f"Rerank error: {e}")
            return unique_docs[: self.top_n]


# =============================================================================
# DATA LOADING & DATABASE SYNC
# =============================================================================
# المسؤولين عن تحميل الـ JSON وربطه بالـ Chroma vector database.
#
# load_docs(file_path):
#   بيقرأ الـ JSON النهائي ويحوله لـ LangChain Document objects.
#   كل document عنده:
#     - page_content : النص اللي هيتعمله embedding وبحث
#     - metadata     : domain / type / display_hint / إلخ + doc_id
#
# _content_hash(text):
#   بيحسب MD5 hash للنص — بيتستخدم في الـ sync
#   عشان نعرف لو الـ document اتغير من آخر مرة.
#
# sync_database(raw_docs, vectorstore):
#   الدالة دي ذكية — بدل ما تمسح وتعيد الـ DB من الأول كل مرة،
#   بتعمل 3-way diff بين الـ JSON وبين اللي موجود في Chroma:
#     - to_insert : documents جديدة مش موجودة في Chroma → تتضاف
#     - to_delete : documents اتحذفت من الـ JSON → تتمسح من Chroma
#     - to_update : documents اتغير نصها → تتمسح وتتضاف تاني
#   الـ batching (batch_size=32) عشان ما يحملش الذاكرة دفعة واحدة.
# =============================================================================
def load_docs(file_path: str) -> tuple[list[dict], list[Document]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_docs: list[dict] = json.load(f)

    lc_docs = [
        Document(
            page_content=d["text"],
            metadata={**d["metadata"], "doc_id": d["id"]},
        )
        for d in raw_docs
    ]
    log.info(f"Loaded {len(lc_docs)} docs from {file_path}")
    return raw_docs, lc_docs


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def sync_database(raw_docs: list[dict], vectorstore: Chroma) -> None:
    existing = vectorstore._collection.get(include=["metadatas"])
    id_to_chroma: dict[str, str] = {}
    id_to_hash: dict[str, str] = {}
    for chroma_id, meta in zip(existing["ids"], existing["metadatas"]):
        doc_id = meta.get("doc_id", chroma_id)
        id_to_chroma[doc_id] = chroma_id
        id_to_hash[doc_id] = meta.get("_hash", "")

    new_map = {d["id"]: d for d in raw_docs}
    to_insert = set(new_map) - set(id_to_chroma)
    to_delete = set(id_to_chroma) - set(new_map)
    to_update = {
        doc_id for doc_id in new_map
        if doc_id in id_to_hash
           and _content_hash(new_map[doc_id]["text"]) != id_to_hash[doc_id]
    }

    if to_delete:
        chroma_ids = [id_to_chroma[doc_id] for doc_id in to_delete]
        vectorstore._collection.delete(ids=chroma_ids)
        log.info(f"Deleted {len(chroma_ids)} stale docs")

    if to_update:
        chroma_ids = [id_to_chroma[doc_id] for doc_id in to_update]
        vectorstore._collection.delete(ids=chroma_ids)
        to_insert |= to_update
        log.info(f"Re-inserting {len(to_update)} updated docs")

    if to_insert:
        batch = [new_map[i] for i in to_insert]
        log.info(f"Inserting {len(batch)} docs...")
        for i in range(0, len(batch), cfg.batch_size):
            b = batch[i: i + cfg.batch_size]
            lc_b = [
                Document(
                    page_content=d["text"],
                    metadata={
                        **d["metadata"],
                        "doc_id": d["id"],
                        "_hash": _content_hash(d["text"]),
                    },
                )
                for d in b
            ]
            vectorstore.add_documents(lc_b)

    if not to_insert and not to_delete and not to_update:
        log.info("DB already up to date — no sync needed")
    else:
        log.info("DB sync complete ✅")


# =============================================================================
# RAG PIPELINE — القلب التقني للنظام
# =============================================================================
# الكلاس ده بيجمع كل أدوات الاسترجاع في pipeline واحد متكامل.
#
# __init__ (التهيئة):
#   بيعمل 4 حاجات عند أول تشغيل:
#     1. بيحمّل موديل الـ Embeddings (BGE-M3) على الـ CPU
#     2. بيفتح أو بيعمل Chroma DB ويعمل sync مع الـ JSON
#     3. بيبني BM25 index على كل الـ documents (global fallback)
#     4. بيحمّل الـ FlagReranker
#
# _build_retriever (بناء الـ Retriever الديناميكي):
#   بيبني retriever مخصص لكل سؤال بناءً على الـ metadata_filter.
#   لو في filter:
#     - بيضيّق نطاق الـ Chroma search بالـ filter
#     - بيبني BM25 مؤقت على الـ docs المفلترة بس (مش على الكل)
#     - لو الـ filter ما رجعش نتائج → fallback للـ corpus الكامل [FIX-5]
#   لو مفيش filter → بيستخدم الـ BM25 global العام
#   في الآخر:
#     - بيعمل EnsembleRetriever يجمع Dense + BM25 بأوزان (0.4 / 0.6)
#     - بيلفه في ContextualCompressionRetriever اللي بيستخدم الـ Reranker
#
# retrieve (نقطة الدخول الرئيسية للبحث):
#   1. بيشيل intent من السؤال (strip_intent)
#   2. بيحدد الـ metadata filter المناسب (get_metadata_filter)
#   3. بيبني الـ retriever المناسب ويشغله
#   4. بيرجع الـ docs النهائية بعد الـ reranking
# =============================================================================
class RAGPipeline:
    def __init__(self, raw_docs: list[dict], lc_docs: list[Document]):
        self._lc_docs = lc_docs

        embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Embeddings loaded ✅")

        self._vectorstore = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embeddings,
            persist_directory=cfg.chroma_path,
            collection_metadata={"hnsw:space": "cosine"},
        )
        sync_database(raw_docs, self._vectorstore)

        self._bm25_global = BM25Retriever.from_documents(
            lc_docs,
            preprocess_func=tokenize,
            k=cfg.top_k_retrieve,
        )

        self._reranker = FlagRerankerCompressor(
            model_name=cfg.rerank_model,
            top_n=cfg.top_k_final,
        )

    def _build_retriever(self, metadata_filter: Optional[dict] = None):
        search_kwargs: dict = {"k": cfg.top_k_retrieve}

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
            filter_key = list(metadata_filter.keys())[0]
            filter_val = metadata_filter[filter_key]

            filtered_docs = [
                d for d in self._lc_docs
                if d.metadata.get(filter_key) == filter_val
            ]
            # [FIX-5] Graceful fallback: if new filter field (sub_domain/payment_type)
            # returns empty (e.g. Chroma doesn't index custom fields the same way),
            # fall back to full corpus to avoid silent empty results
            if not filtered_docs:
                log.warning(
                    f"Filter '{filter_key}={filter_val}' returned 0 docs — "
                    f"falling back to full corpus"
                )
                filtered_docs = self._lc_docs
                search_kwargs.pop("filter", None)  # also remove from vectorstore filter

            active_bm25 = BM25Retriever.from_documents(
                filtered_docs,
                preprocess_func=tokenize,
                k=cfg.top_k_retrieve,
            )
        else:
            active_bm25 = self._bm25_global

        dense = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        ensemble = EnsembleRetriever(
            retrievers=[dense, active_bm25],
            weights=[cfg.dense_weight, cfg.bm25_weight],
            c=60,
        )
        return ContextualCompressionRetriever(
            base_compressor=self._reranker,
            base_retriever=ensemble,
        )

    def retrieve(self, query: str) -> list[Document]:
        retrieval_query = strip_intent(query)
        if retrieval_query != query:
            log.info(f"Intent stripped: '{query}' → '{retrieval_query}'")

        metadata_filter = get_metadata_filter(retrieval_query)
        if metadata_filter:
            log.info(f"Metadata filter applied: {metadata_filter}")

        retriever = self._build_retriever(metadata_filter)
        docs = retriever.invoke(retrieval_query)

        log.info(f"Retrieved {len(docs)} chunks for: '{retrieval_query}'")
        for i, doc in enumerate(docs):
            log.info(f"  [{i}] [{doc.metadata.get('type', '?')}] {doc.page_content[:80]}...")

        return docs


# =============================================================================
# WE ASSISTANT — الواجهة الكاملة للمحادثة
# =============================================================================
# الكلاس ده بيمثل الـ chatbot اللي العميل بيتكلم معاه.
# بيربط بين الـ RAGPipeline والـ LLM ويدير المحادثة كاملة.
#
# SYSTEM_PROMPT:
#   التعليمات الثابتة اللي بتتبعت للـ LLM في كل رسالة.
#   بتحدد شخصية WE-Bot، قواعد اللغة، القيود، وتنسيق الردود.
#   مش بتتغير — بس ممكن يتضاف عليها override ديناميكي لو العميل بيتكلم إنجليزي.
#
# __init__:
#   بيربط الـ RAGPipeline ويهيئ الـ LLM (GPT-4o-mini عبر OpenRouter)
#   وبيعمل history list فاضية للمحادثة.
#
# _condense_query (تحويل السؤال لسؤال مستقل):
#   لما العميل يسأل سؤال غامض أو متابعة (مثل "وبكام؟")،
#   الدالة دي بتبعت آخر 6 رسائل من التاريخ + السؤال الجديد للـ LLM
#   وبتطلب منه يصيغ سؤال مستقل بالعربي مناسب للبحث في الـ DB.
#
# chat (الدالة الرئيسية — نقطة دخول كل رسالة):
#   الـ flow الكامل لكل رسالة:
#     1. بيكشف لغة العميل (عربي / إنجليزي) → يضبط الـ system prompt
#     2. لو تحية → يتخطى الـ RAG ويرد مباشرة (توفير tokens)
#     3. لو سؤال حقيقي:
#        أ. _condense_query: يصيغ السؤال للبحث
#        ب. rag.retrieve: يجيب الـ docs الأنسب
#        ج. يفرمت الـ chunks مع headers (domain / sub_domain / display_hint)
#           [FIX-6] اللي أضاف sub_domain و payment_type في الـ header
#     4. يبني الـ messages list (system + history + context + question)
#     5. يبعت للـ LLM ويرجع الرد
#     6. يحفظ السؤال والرد في الـ history للاستخدام في الأسئلة الجاية
# =============================================================================
class WEAssistant:
    SYSTEM_PROMPT = """أنت "وي-بوت" (WE-Bot)، المساعد الذكي والممثل الرقمي الرسمي للشركة المصرية للاتصالات (WE). شخصيتك ودودة، عملية، وتعتز بالهوية المصرية.

               ## 1. الهدف الأساسي (The Mission)
               مهمتك هي تقديم دعم دقيق وفوري لعملاء WE بناءً على **السياق (Context)** المقدم فقط. أنت لست مجرد محرك بحث، بل خبير يسهل المعلومة.

               ## 2. ميثاق الإجابة (Response Guidelines)
               - **الدقة المطلقة**: لا تقدم أي أرقام، أسعار، أو أكواد (USSD) غير موجودة نصاً في السياق.
               - **الهوية واللغة**:
                 - **القاعدة الأساسية**: رد دائماً بنفس لغة العميل تماماً — عربي بعربي، إنجليزي بإنجليزي. لا تخلط اللغتين أبداً.
                 - إذا سأل بالعامية المصرية، جاوب بعامية مصرية مهذبة وودودة (مثل: "يا فندم"، "تحت أمرك"، "نورتنا").
                 - إذا سأل بالفصحى، التزم بالفصحى.
                 - إذا سأل بالإنجليزية، جاوب بالإنجليزية الكاملة حتى لو المعلومة في السياق بالعربي — ترجمها أنت.
                 - استخدم المصطلحات التقنية كما هي (مثل: "router"، "MB"، "package" في الإنجليزي / "راوتر"، "ميجا"، "باقة" في العربي).
               - **التعامل مع الفروع**:
                 - لو ذُكرت الفروع بشكل عرضي في أي رد (مثل: "تقدر تشتري من أي فرع WE")، أضف جملة مثل: "لو عايز أقرب فرع ليك، تقدر تزور موقع وي"


               ## 3. القيود الصارمة (Strict Constraints)
               - **منع الهلوسة**: إذا غابت المعلومة تماماً ولم تجد أي شيء متعلق بها في السياق، قل: "بعتذر لك جداً، المعلومة دي مش متوفرة عندي حالياً. تقدر تشرفنا بالاتصال بـ 111 أو 01555000111 أو زيارة موقعنا te.eg". أما إذا وجدت معلومة قريبة أو متعلقة، حاول مساعدة العميل بأقصى قدر ممكن بناءً عليها.
               - **السرية**: لا تذكر أبداً أنك تستخدم "سياق" أو "ملفات"؛ تصرف كأنك تمتلك هذه المعرفة ذاتياً.
               - **خدمة "معاك"**: تذكر أنها للصم وضعاف السمع (فيديو إشارة). لا توجههم لـ 111 أبداً؛ وجههم لموقع te.eg (من 9 ص لـ 9 م).

               ## 4. تنسيق المخرجات (Output Formatting)
               - **القوائم**: إذا كانت الإجابة تحتوي على خطوات أو خيارات، استخدم النقاط (Bullet points) لجعلها سهلة القراءة.
               - **الأكواد**: ضع أكواد الـ USSD (مثل *100#) في تنسيق **Bold** أو `code` لتكون واضحة.
               - **الروابط**: اذكر اسم الصفحة/الخدمة متبوعاً بالرابط الكامل إن وُجد. (مثال: "تقدر تشحن من تطبيق My WE على te.eg").
               - **ممنوع منعاً باتاً**: لا تقول أبداً "اضغط هنا" أو "انقر هنا" أو "من هنا" — أنت chatbot نصي ومفيش أزرار. بدلاً منها اذكر الرابط أو القناة مباشرةً (مثال: بدل "اضغط هنا لمعرفة الفروع" قول "تقدر تعرف أقرب فرع من موقع te.eg أو تطبيق My WE").
               -الاجابة تكون احترافية وليس كل اجاباتك تنتهي ب جمل متكررة (مثل انا هنا للمساعدة) وجمل شبيهة. 

               ## 5. منطق المتابعة (Follow-up Logic)
               - إذا سأل العميل سؤالاً ناقصاً (مثل: "بكام؟" أو "إزاي؟")، ارجع لآخر سياق تم ذكره وجاوب عليه. إذا كان الغموض سيد الموقف، اسأل سؤالاً توضيحياً بذكاء.
               """

    def __init__(self, rag: RAGPipeline):
        self.rag = rag
        self.history: list[BaseMessage] = []
        self.llm = ChatOpenAI(
            model=cfg.llm_model,
            temperature=0.1,
            api_key=cfg.openrouter_key,
            base_url=cfg.llm_base_url,
            timeout=30,
            max_tokens=2000
        )

    def _condense_query(self, query: str) -> str:
        if not self.history or not is_ambiguous(query):
            pass

        history_str = "\n".join(
            f"{'عميل' if isinstance(m, HumanMessage) else 'بوت'}: {m.content}"
            for m in self.history[-cfg.history_condense:]
        )

        prompt = (
            f"بناءً على المحادثة التالية، حوّل سؤال العميل الأخير لسؤال مستقل وواضح.\n"
            f"⚠️ **هام جداً:** يجب صياغة السؤال المستقل باللغة **العربية** دائماً (لأننا سنبحث في قاعدة بيانات عربية)، حتى لو كان سؤال العميل بلغة أخرى.\n"
            f"أعد السؤال المستقل فقط بدون أي كلام زيادة.\n\n"
            f"المحادثة:\n{history_str}\n\n"
            f"سؤال العميل: {query}\n"
            f"السؤال المستقل:"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        standalone = response.content.strip()

        if not standalone or len(standalone) < 3:
            return query

        log.info(f"Search Query (Arabic): '{query}' → '{standalone}'")
        return standalone

    def chat(self, query: str) -> str:
        q_clean = query.strip()

        user_is_arabic = is_arabic(q_clean)

        dynamic_system_prompt = self.SYSTEM_PROMPT
        if not user_is_arabic:
            dynamic_system_prompt += (
                "\n\n[CRITICAL SYSTEM OVERRIDE]: "
                "The current user is communicating in ENGLISH. "
                "You MUST translate all retrieved WE context into English and respond ONLY in English. "
                "Disregard the Arabic persona examples completely for this specific interaction."
            )

        if _GREETINGS_PATTERN.match(q_clean):
            log.info(f"Greeting detected: '{query}' → Skipping RAG retrieval to optimize tokens.")
            if user_is_arabic:
                context_text = "No context needed. The user is just greeting you. Reply warmly in Egyptian Arabic."
            else:
                context_text = (
                    "No context needed. The user is just greeting you. "
                    "CRITICAL: You MUST reply to the greeting ENTIRELY in English "
                    "(e.g., 'Hello! I am WE-Bot. How can I assist you today?')."
                )
        else:
            search_query = self._condense_query(query)
            docs = self.rag.retrieve(search_query)

            formatted_chunks = []
            for i, doc in enumerate(docs):
                hint = doc.metadata.get("display_hint", "normal")
                domain = doc.metadata.get("domain", "general")
                # [FIX-6] Include sub_domain and payment_type in context header when present
                sub_domain = doc.metadata.get("sub_domain", "")
                payment_type = doc.metadata.get("payment_type", "")
                extra = ""
                if sub_domain:
                    extra += f" | Sub-Domain: {sub_domain}"
                if payment_type:
                    extra += f" | Payment Type: {payment_type}"
                chunk_str = (
                    f"[Context Segment {i + 1} | Domain: {domain}{extra} | Output Constraint: {hint}]\n"
                    f"{doc.page_content}"
                )
                formatted_chunks.append(chunk_str)

            context_text = "\n\n---\n\n".join(formatted_chunks)

        instruction = (
            "أجب على السؤال التالي بناءً على السياق أعلاه باللغة العربية."
            if user_is_arabic
            else "CRITICAL: You MUST answer the Question below ENTIRELY in ENGLISH based on the Context provided. DO NOT output Arabic."
        )

        messages: list[BaseMessage] = [
            SystemMessage(content=dynamic_system_prompt),
            *self.history[-cfg.history_llm:],
            HumanMessage(
                content=f"Context:\n{context_text}\n\n{instruction}\n\nQuestion: {query}"
            ),
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=answer))

        return answer


# =============================================================================
# MAIN — نقطة تشغيل البرنامج
# =============================================================================
# بيعمل الخطوات دي بالترتيب:
#   1. يحمّل الـ documents من الـ JSON
#   2. يبني الـ RAGPipeline (embeddings + chroma + bm25 + reranker)
#   3. يبني الـ WEAssistant (llm + history)
#   4. يدخل في loop تفاعلي:
#      - يقرأ input من العميل
#      - لو كلمة خروج (exit / شكرا / إلخ) → ينهي
#      - غير كده → يبعت للـ bot ويطبع الرد
#      - لو حصل خطأ غير متوقع → يطبع رسالة خطأ ويكمل
# =============================================================================
if __name__ == "__main__":
    raw_docs, lc_docs = load_docs(cfg.data_file)
    rag = RAGPipeline(raw_docs, lc_docs)
    bot = WEAssistant(rag)

    print("\n" + "=" * 50)
    print("🚀 WE Intelligent Assistant (LangChain) is Ready!")
    print("Type your message to start chatting.")
    print("Type 'exit' or 'خروج' to end.")
    print("=" * 50 + "\n")

    EXIT_TRIGGERS = {"شكرا", "تمام", "شكرًا", "exit", "quit", "خروج", "stop"}

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in EXIT_TRIGGERS:
                print("\nAssistant: شكراً لتواصلك مع WE. يومك سعيد!")
                break

            response = bot.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: شكراً لتواصلك مع WE. يومك سعيد!")
            break
        except Exception as e:
            log.error(f"Chat error: {e}")
            print("\nAssistant: عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.")
