"""
WE RAG — Data Layer Fixes
=========================
يعالج المشاكل الـ 3 من جذرها في الداتا نفسها قبل ما تتحط في ChromaDB

المشكلة 1: URLs/روابط بتخلي المودل يقول "اضغط هنا"
المشكلة 2: Chunks معزولة عن context بتاعها (زي سؤال تفعيل الخط)
المشكلة 3: metadata مش بتفرق بين الشركة والخدمات
"""

import json
import re
import os
import uuid
from pathlib import Path

# ══════════════════════════════════════════════
# FIX 1: URL/Link Cleaner
# بيحول الـ URLs لنص وصفي بدل روابط خام
# ══════════════════════════════════════════════

URL_REPLACEMENTS = {
    # روابط معروفة نعرف معناها
    "https://te.eg/about-te/Ma3ak":
        "خدمة معاك متاحة على الموقع الرسمي te.eg في قسم معاك",
    "https://maps.app.goo.gl/tZ7iF6K6bZqFv7jQ8":
        "الموقع متاح على خرائط جوجل",
    "https://www.te.eg":
        "الموقع الرسمي للمصرية للاتصالات te.eg",
    "https://apps.apple.com/eg/app/my-we/id1413151505":
        "تطبيق My WE متاح على App Store لأجهزة iOS",
    "https://play.google.com/store/apps/details?id=com.ucare.we":
        "تطبيق My WE متاح على Google Play لأجهزة Android",
    "https://www.facebook.com/TelecomEgypt":
        "صفحة وي على فيسبوك: TelecomEgypt",
    "https://x.com/telecomegypt":
        "حساب وي على تويتر/X: telecomegypt",
    "https://www.youtube.com/channel/UCLl_SOH0KD8-Hv1H1yetEyw":
        "قناة وي على يوتيوب",
    "https://www.instagram.com/telecom.egypt/":
        "حساب وي على انستجرام: telecom.egypt",
    "https://www.linkedin.com/company/telecom-egypt/":
        "صفحة وي على لينكدإن: telecom-egypt",
}

# Pattern للروابط اللي مش في القاموس فوق
URL_PATTERN = re.compile(r'https?://\S+')

def clean_urls(text: str) -> str:
    """بيستبدل الـ URLs بنص وصفي."""
    for url, replacement in URL_REPLACEMENTS.items():
        text = text.replace(url, replacement)
    # أي URL باقي → احذفه بهدوء
    text = URL_PATTERN.sub("", text)
    # نضيف نص توجيه عام لو كان في الأصل رابط
    return text.strip()


# ══════════════════════════════════════════════
# FIX 2: Context-Aware Chunking
# بيضم الـ chunks الصغيرة المعزولة مع parent context بتاعها
# ══════════════════════════════════════════════

# الـ types دي لما بتيجي لوحدها بتكون confusing
# محتاجة تتدمج مع الـ parent chunk
ORPHAN_TYPES = {
    "line_activation",      # سؤال تفعيل الخط
    "definition",           # تعريفات العقد
    "prohibition",          # محظورات منفردة
    "force_majeure",        # أسباب القوة القاهرة
    "iptv_restriction",     # قيود IPTV
    "timeline_event",       # أحداث تاريخية منفردة
    "board_member",         # أعضاء مجلس الإدارة
    "executive_management", # الإدارة التنفيذية
    "hotline",              # أرقام خطوط منفردة
    "email",                # إيميلات منفردة
}

def group_orphan_chunks(docs: list[dict]) -> list[dict]:
    """
    بيجمع الـ chunks الصغيرة المتشابهة في chunk واحدة كبيرة.
    مثلاً: كل أعضاء مجلس الإدارة → chunk واحدة.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    standalone = []

    for doc in docs:
        t = doc["metadata"].get("type", "")
        src = doc["metadata"].get("_source_file", "unknown")
        if t in ORPHAN_TYPES:
            groups[(src, t)].append(doc)
        else:
            standalone.append(doc)

    merged = []
    for (src, t), group in groups.items():
        # رتّب بالـ index لو موجود
        group.sort(key=lambda x: x["metadata"].get("index", 0))

        combined_text = "\n".join(d["text"] for d in group)
        new_doc = {
            "id": str(uuid.uuid4()),
            "text": combined_text,
            "metadata": {
                "type": t,
                "_source_file": src,
                "_merged": True,
                "_count": len(group)
            }
        }
        merged.append(new_doc)

    return standalone + merged


# ══════════════════════════════════════════════
# FIX 3: Metadata Enrichment
# بيضيف domain + entity_type لكل chunk
# ══════════════════════════════════════════════

# ── Mapping: source file → domain label ──────
SOURCE_DOMAIN_MAP = {
    "normalized_aboutwe.json":          "company_info",
    "normalized_B2C_Contract.json":     "contracts_legal",
    "normalized_bill_service.json":     "billing",
    "normalized_branches.json":         "branches",
    "normalized_customer_support.json": "support",
    "normalized_faq.json":              "faq",
    "normalized_important_numbers.json":"ussd_codes",
    "normalized_ma3ak_service.json":    "accessibility_service",
    "normalized_payment_methods.json":  "payments",
    "normalized_we_bonus.json":         "offers_promotions",
    "normalized140dlel.json":           "directory_service",
}

# ── الـ types اللي بتتكلم عن الشركة نفسها ────
COMPANY_ENTITY_TYPES = {
    "company_overview", "achievement", "timeline_event",
    "museum_info", "hq_location", "official_links",
    "board_member", "executive_management"
}

# ── الـ types اللي بتتكلم عن خدمة/باقة ───────
SERVICE_ENTITY_TYPES = {
    "service_overview", "service_code", "pricing",
    "activation", "subscription", "requirements",
    "supported_browsers", "working_hours", "payment_method",
    "directory_service"
}

def enrich_metadata(doc: dict, source_file: str) -> dict:
    """بيضيف domain + entity_type + display_hint للـ metadata."""
    meta = doc.get("metadata", {})
    doc_type = meta.get("type", "")

    # domain
    meta["domain"] = SOURCE_DOMAIN_MAP.get(source_file, "general")

    # entity_type — ده اللي بيحل مشكلة خلط الشركة بالخدمات
    if doc_type in COMPANY_ENTITY_TYPES:
        meta["entity_type"] = "company"
    elif doc_type in SERVICE_ENTITY_TYPES:
        meta["entity_type"] = "service"
    elif meta["domain"] == "contracts_legal":
        meta["entity_type"] = "legal"
    else:
        meta["entity_type"] = "general"

    # display_hint — بيوجّه المودل إزاي يعرض المعلومات
    if doc_type in ("official_links", "hq_location", "hotline"):
        meta["display_hint"] = "provide_text_only_no_clickable_links"
    elif doc_type == "line_activation":
        meta["display_hint"] = "security_question_context"
    else:
        meta["display_hint"] = "normal"

    meta["_source_file"] = source_file
    doc["metadata"] = meta
    return doc


# ══════════════════════════════════════════════
# PIPELINE: تطبيق كل الـ fixes على الداتا
# ══════════════════════════════════════════════

def process_file(file_path: str) -> list[dict]:
    """بيقرأ JSON واحد ويطبق عليه كل الـ fixes."""
    source_file = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if isinstance(docs, dict):
        docs = [docs]

    processed = []
    for doc in docs:
        # Fix 1: نظّف الـ URLs
        doc["text"] = clean_urls(doc["text"])

        # Fix 3: أضف metadata غنية
        doc = enrich_metadata(doc, source_file)

        processed.append(doc)

    return processed


def process_all(input_folder: str, output_folder: str):
    """بيعالج كل الـ JSON files وبيحفظهم في فولدر جديد."""
    os.makedirs(output_folder, exist_ok=True)
    all_docs = []

    for fname in os.listdir(input_folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(input_folder, fname)
        docs  = process_file(fpath)
        all_docs.extend(docs)
        print(f"  ✅ {fname}: {len(docs)} chunks")

    # Fix 2: ادمج الـ orphan chunks بعد ما نجمع كل الداتا
    print(f"\n📦 قبل الدمج: {len(all_docs)} chunks")
    all_docs = group_orphan_chunks(all_docs)
    print(f"📦 بعد الدمج: {len(all_docs)} chunks")

    # احفظ ملف موحّد للـ ChromaDB
    out_path = os.path.join(output_folder, "processed_all.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"\n💾 تم الحفظ في: {out_path}")
    return all_docs


# ══════════════════════════════════════════════
# RETRIEVAL FIX: Metadata-Aware Search
# بيضيف entity_type filter للـ query لما يلزم
# ══════════════════════════════════════════════

# كلمات دلالة → entity_type filter
QUERY_ENTITY_HINTS = {
    "company":  ["شركة", "تاريخ", "تأسيس", "مجلس الإدارة", "رئيس", "رؤية",
                 "about", "history", "founded", "CEO", "إنجاز", "1854"],
    "service":  ["خدمة", "باقة", "اشتراك", "تفعيل", "أسعار", "رصيد",
                 "service", "plan", "offer", "subscribe", "معاك", "بونص"],
    "legal":    ["عقد", "شروط", "حقوق", "التزام", "غرامة", "contract", "terms"],
    "billing":  ["فاتورة", "سداد", "دفع", "bill", "payment", "invoice"],
}

def detect_entity_type(query: str) -> str | None:
    """بيكشف من السؤال إيه الـ entity_type المطلوب."""
    q_lower = query.lower()
    for entity_type, keywords in QUERY_ENTITY_HINTS.items():
        if any(kw in q_lower for kw in keywords):
            return entity_type
    return None  # مش واضح → متفلترش


def smart_retrieve(collection, embed_model, reranker, query: str,
                   top_k_retrieve: int = 10,
                   top_k_final: int = 4,
                   score_threshold: float = 0.35) -> list[dict]:
    """
    Retrieval محسّن بـ:
    1. entity_type filter تلقائي
    2. distance threshold
    3. cross-encoder re-ranking
    """
    # Detect entity type from query
    entity_type = detect_entity_type(query)

    query_vec = embed_model.encode([query]).tolist()

    # Build where filter
    where = None
    if entity_type:
        where = {"entity_type": {"$eq": entity_type}}

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k_retrieve,
        include=["documents", "metadatas", "distances"],
        where=where
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Filter by threshold
    filtered = [
        {"text": d, "meta": m, "dist": dist}
        for d, m, dist in zip(docs, metas, distances)
        if dist <= score_threshold
    ]

    # Fallback: لو مفيش نتايج بعد الفلتر
    if not filtered:
        filtered = [
            {"text": d, "meta": m, "dist": dist}
            for d, m, dist in zip(docs[:2], metas[:2], distances[:2])
        ]

    # Re-rank
    if len(filtered) > 1:
        pairs  = [(query, item["text"]) for item in filtered]
        scores = reranker.predict(pairs)
        for item, score in zip(filtered, scores):
            item["rerank_score"] = float(score)
        filtered.sort(key=lambda x: x["rerank_score"], reverse=True)

    return filtered[:top_k_final]


# ══════════════════════════════════════════════
# PROMPT FIX: display_hint aware system prompt
# ══════════════════════════════════════════════

SYSTEM_PROMPT_V2 = """أنت مساعد ذكي لخدمة عملاء WE (المصرية للاتصالات).

قواعد أساسية:
- أجب فقط بناءً على المعلومات الموجودة في الـ Context.
- لا تخترع أو تفترض أي معلومة غير موجودة.
- إذا لم تجد إجابة واضحة، حوّل العميل لخدمة العملاء: 111 أو 01555000111
- جاوب بنفس لغة المستخدم (مصري / فصحى / إنجليزي).

تعليمات مهمة للعرض:
- إذا وجدت معلومات عن روابط أو مواقع، اذكرها كنص وصفي فقط (مثال: "متاح على الموقع الرسمي te.eg")، ولا تقل "اضغط هنا" أو "انقر هنا".
- إذا كان الـ Context يحتوي على سؤال أمان (مثل "اسم الجد من الأم")، وضّح للعميل أن هذا سؤال تفعيل الخط ولا تعرضه كمعلومة مستقلة.
- لما تتكلم عن شركة WE، ركّز على المعلومات المؤسسية.
- لما تتكلم عن خدمة معاك، وضّح أنها خدمة لغة الإشارة للصم وضعاف السمع."""


if __name__ == "__main__":
    # ── للتجربة: غيّر المسارات دي ──
    INPUT_FOLDER  = r"./NormalizedDatav2"
    OUTPUT_FOLDER = r"./ProcessedData"

    print("🚀 بدء معالجة الداتا...\n")
    docs = process_all(INPUT_FOLDER, OUTPUT_FOLDER)

    # عرض sample من النتيجة
    print("\n── Sample chunk بعد المعالجة ──")
    for doc in docs[:2]:
        print(f"ID     : {doc['id']}")
        print(f"Domain : {doc['metadata'].get('domain')}")
        print(f"Entity : {doc['metadata'].get('entity_type')}")
        print(f"Hint   : {doc['metadata'].get('display_hint')}")
        print(f"Text   : {doc['text'][:120]}...")
        print("─" * 50)
