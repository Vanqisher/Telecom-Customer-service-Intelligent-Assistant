import streamlit as st
import time
from LangChainV import WEAssistant, load_docs, RAGPipeline, cfg
import base64
from pypdf import PdfReader
import docx2txt
import easyocr

@st.cache_resource
def load_ocr_model():
    # 'gpu=False' forces it to use CPU, which is more stable on most Windows setups
    return easyocr.Reader(['ar', 'en'], gpu=False)

# Store the reader in session_state so it stays loaded
if "ocr_reader" not in st.session_state:
    st.session_state.ocr_reader = load_ocr_model()
def extract_text_from_image(uploaded_file):
    try:
        # EasyOCR works directly with the file bytes
        reader = st.session_state.ocr_reader
        text_results = reader.readtext(uploaded_file.getvalue(), detail=0)

        # Combine the list of detected strings into one block
        text = "\n".join(text_results)

        # ANSI escape codes for red text in the terminal
        RED = '\033[91m'
        RESET = '\033[0m'

        print(f"{RED}\n" + "=" * 40)
        print("🛑 EASYOCR TERMINAL TEST OUTPUT:")

        if not text.strip():
            print("[NO TEXT RECOGNIZED BY EASYOCR]")
        else:
            print(text)

        print("=" * 40 + f"{RESET}\n")

        return text.strip()

    except Exception as e:
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f"{RED}\n❌ EASYOCR ERROR: {e}\n{RESET}")

        st.error(f"EasyOCR error: {e}")
        return ""

        # ANSI escape codes for red text in the terminal
        RED = '\033[91m'
        RESET = '\033[0m'

        print(f"{RED}\n" + "=" * 40)
        print("🛑 OCR TERMINAL TEST OUTPUT:")

        if not text.strip():
            print("[NO TEXT RECOGNIZED BY TESSERACT]")
        else:
            print(text)

        print("=" * 40 + f"{RESET}\n")

        return text.strip()

    except Exception as e:
        # Also print the error in red in the terminal
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f"{RED}\n❌ OCR ERROR: {e}\n{RESET}")

        st.error(f"OCR error: {e}")
        return ""

    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""


def extract_text_from_file(uploaded_file):
    """دالة لقراءة الملفات المرفوعة وتحويلها إلى نص صافي"""
    file_name = uploaded_file.name.lower()

    try:
        # 1. ملفات التكست العادية
        if file_name.endswith('.txt'):
            return uploaded_file.read().decode("utf-8")

        # 2. ملفات الـ PDF
        elif file_name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

        # 3. ملفات الوورد (DOCX)
        elif file_name.endswith('.docx'):
            return docx2txt.process(uploaded_file)

        return None
    except Exception as e:
        st.error(f"حصلت مشكلة وأنا بقرأ الملف: {e}")
        return None


def escape_markdown(text):
    # Fixed: Removed the aggressive escaping that was breaking links and bold text
    # Only escape characters that truly break Streamlit's markdown rendering if necessary
    return text


def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""


IMAGE_PATH = "We_logo.svg.png"
LOGO_B64 = get_image_base64(IMAGE_PATH)

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WE Intelligent Assistant",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 3. CUSTOM CSS
# ─────────────────────────────────────────────
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --we-purple: #5b0fa8;
    --we-purple-light: #7b2fd4;
    --we-purple-dark: #3a0870;
    --we-accent: #9b59f7;
    --we-glow: rgba(91, 15, 168, 0.35);
    --sidebar-bg: #ffffff;
    --sidebar-border: #ede8f7;
    --chat-bg: #f9f7fd;
    --user-bubble: #ede8f7;
    --assistant-bubble: #ffffff;
    --text-dark: #1a0535;
    --text-mid: #4a3060;
    --text-light: #8a77a8;
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
    --shadow-card: 0 4px 24px rgba(91,15,168,0.10);
    --shadow-glow: 0 0 40px rgba(91,15,168,0.18);
}

/* ── GLOBAL RESET ── */
html, body, .stApp {
    font-family: 'Outfit', 'Cairo', sans-serif !important;
    background: #0d0221 !important;
}

/* ── MAIN BACKGROUND: Dark purple galaxy ── */
.stApp {
    background: linear-gradient(135deg, #0d0221 0%, #1a0840 40%, #0d0221 100%) !important;
    background-attachment: fixed !important;
}

/* Starfield overlay via pseudo-element workaround */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* ── HEADER ── */
.we-header {
    background: linear-gradient(135deg, #1a0840 0%, #2d0d6e 50%, #1a0840 100%);
    border-bottom: 1px solid rgba(155, 89, 247, 0.25);
    padding: 18px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
    margin-bottom: 0;
}
.we-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(155,89,247,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.we-header::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 120px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(91,15,168,0.20) 0%, transparent 70%);
    pointer-events: none;
}
.we-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.we-header-logo {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    object-fit: contain;
    background: rgba(255,255,255,0.07);
    padding: 6px;
    border: 1px solid rgba(155,89,247,0.30);
}
.we-header-title {
    margin: 0;
    font-family: 'Outfit', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
    line-height: 1.1;
}
.we-header-subtitle {
    margin: 0;
    font-family: 'Cairo', sans-serif;
    font-size: 0.9rem;
    color: rgba(200,180,240,0.85);
    font-weight: 400;
    direction: rtl;
}
.we-header-badge {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    box-shadow: 0 2px 12px rgba(91,15,168,0.4);
}

/* ── CHAT AREA WRAPPER ── */
.main-chat-wrapper {
    background: #f9f7fd;
    margin: 0;
    min-height: calc(100vh - 120px);
    border-radius: 0;
    padding-bottom: 180px;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 24px !important;
    direction: rtl;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: linear-gradient(135deg, #ede8f7, #e4dbf5) !important;
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg) !important;
    padding: 14px 18px !important;
    color: var(--text-dark) !important;
    font-size: 0.95rem;
    line-height: 1.7;
    box-shadow: 0 2px 10px rgba(91,15,168,0.08);
    border: 1px solid rgba(91,15,168,0.10);
    direction: rtl;
    text-align: right;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) .stMarkdown {
    background: #ffffff !important;
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px !important;
    padding: 14px 18px !important;
    color: var(--text-dark) !important;
    font-size: 0.95rem;
    line-height: 1.75;
    box-shadow: 0 2px 16px rgba(91,15,168,0.07);
    border: 1px solid rgba(91,15,168,0.08);
    direction: rtl;
    text-align: right;
}

/* Message text & Lists */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol {
    font-family: 'Cairo', 'Outfit', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    margin-bottom: 6px;
}
.stMarkdown ol, .stMarkdown ul {
    padding-right: 25px !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7) !important;
    border: 2px solid rgba(155,89,247,0.4) !important;
    box-shadow: 0 2px 12px rgba(91,15,168,0.3) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: #1a0840 !important;
    border-top: 1px solid rgba(155,89,247,0.25) !important;
    padding: 10px 18px !important;
    min-height: 70px !important;
}

[data-testid="stChatInputTextArea"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(155,89,247,0.35) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.92rem !important;
    direction: rtl !important;
    text-align: right !important;
    padding: 10px 16px !important;
    min-height: 20px !important;
    max-height: 90px !important;
    line-height: 1.4 !important;
    overflow-y: auto !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: #9b59f7 !important;
    box-shadow: 0 0 0 3px rgba(155,89,247,0.20) !important;
    background: rgba(255,255,255,0.09) !important;
    outline: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder {
    color: rgba(200,180,240,0.45) !important;
    font-family: 'Cairo', sans-serif !important;
}

/* Submit button */
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7) !important;
    border: none !important;
    border-radius: 12px !important;
    width: 44px !important;
    height: 44px !important;
    box-shadow: 0 4px 16px rgba(91,15,168,0.5) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: linear-gradient(135deg, #7b2fd4, #b07df8) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 6px 20px rgba(91,15,168,0.6) !important;
}
[data-testid="stChatInputSubmitButton"] button svg {
    fill: white !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--sidebar-border) !important;
    box-shadow: 4px 0 24px rgba(91,15,168,0.06) !important;
}
[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

/* Sidebar content wrapper */
.sidebar-content {
    padding: 24px 20px;
    direction: rtl;
}

/* Sidebar logo area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 20px 16px;
    border-bottom: 1px solid var(--sidebar-border);
    margin-bottom: 20px;
}
.sidebar-brand img {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    object-fit: contain;
    background: #f3eeff;
    padding: 5px;
    border: 1px solid rgba(91,15,168,0.12);
}
.sidebar-brand-text {
    font-family: 'Cairo', sans-serif;
    font-size: 0.8rem;
    color: var(--text-mid);
    direction: rtl;
    text-align: right;
    line-height: 1.3;
}
.sidebar-brand-text strong {
    font-size: 0.95rem;
    color: var(--we-purple);
    display: block;
}

/* Section header */
.sidebar-section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-light);
    padding: 0 20px;
    margin-bottom: 8px;
    margin-top: 20px;
}

/* Feature items */
.feature-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 20px;
    border-radius: 10px;
    margin: 2px 12px;
    transition: all 0.2s ease;
    cursor: default;
    direction: rtl;
}
.feature-item:hover {
    background: #f3eeff;
}
.feature-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #f3eeff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    border: 1px solid rgba(91,15,168,0.10);
}
.feature-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: var(--text-mid);
    font-weight: 500;
}

/* Support card */
.support-card {
    background: linear-gradient(135deg, #f3eeff, #ede8f7);
    border: 1px solid rgba(91,15,168,0.15);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    margin: 20px 16px;
    direction: rtl;
    text-align: right;
}
.support-card .label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-light);
    margin-bottom: 4px;
}
.support-card .title {
    font-family: 'Cairo', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 10px;
}
.support-card .subtitle {
    font-size: 0.75rem;
    color: var(--text-mid);
    margin-bottom: 6px;
    font-family: 'Outfit', sans-serif;
}
.support-card .number {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--we-purple);
    line-height: 1;
}

/* ── STREAMLIT BUTTONS ── */
.stButton > button {
    width: calc(100% - 32px) !important;
    margin: 0 16px !important;
    background: transparent !important;
    border: 1.5px solid rgba(91,15,168,0.4) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--we-purple) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 16px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 4px 16px rgba(91,15,168,0.30) !important;
    transform: translateY(-1px) !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    color: var(--we-accent) !important;
}

/* ── HIDE BRANDING ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(91,15,168,0.25);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(91,15,168,0.45);
}

/* ── WELCOME MESSAGE ── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 40px 40px;
    text-align: center;
    gap: 16px;
}
.welcome-icon {
    width: 80px;
    height: 80px;
    border-radius: 20px;
    object-fit: contain;
    background: linear-gradient(135deg, #f3eeff, #ede8f7);
    padding: 10px;
    border: 1px solid rgba(91,15,168,0.12);
    box-shadow: 0 8px 32px rgba(91,15,168,0.12);
    margin-bottom: 8px;
}
.welcome-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin: 0;
}
.welcome-sub {
    font-family: 'Cairo', sans-serif;
    font-size: 1rem;
    color: rgba(200,180,240,0.85);
    margin: 0;
    direction: rtl;
}
.welcome-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-top: 16px;
}
.chip {
    background: rgba(155,89,247,0.2);
    border: 1px solid rgba(155,89,247,0.4);
    color: white;
    font-family: 'Cairo', sans-serif;
    font-size: 0.82rem;
    padding: 7px 16px;
    border-radius: 20px;
    cursor: default;
    transition: all 0.2s ease;
}
.chip:hover {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7);
    color: white;
    border-color: transparent;
}

/* ── TIMESTAMP ── */
.msg-time {
    font-size: 0.65rem;
    color: var(--text-light);
    text-align: left;
    direction: ltr;
    margin-top: 4px;
    padding: 0 4px;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1px solid var(--sidebar-border) !important;
    margin: 12px 16px !important;
}
/* FIX CHAT HIDDEN BEHIND INPUT */
.main .block-container {
    padding-bottom: 180px !important;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* ── UPLOAD BUTTONS FIX ── */

div[data-testid="stFileUploader"] {
    width: 100% !important;
}

div[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #5b0fa8, #9b59f7) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.8rem !important;
    padding: 7px 14px !important;
}

/* hide uploaded filename hyperlink */
div[data-testid="stFileUploader"] small {
    display: none !important;
}

div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
    display: none !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. SESSION STATE & INITIALIZATION
# ─────────────────────────────────────────────
@st.cache_resource
def initialize_assistant():
    # load docs
    raw_docs, lc_docs = load_docs(cfg.data_file)
    # create rag pipeline
    rag_pipeline = RAGPipeline(raw_docs, lc_docs)
    # create assistant
    return WEAssistant(rag_pipeline)


if "assistant" not in st.session_state:
    st.session_state.assistant = initialize_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# ─────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo + Brand
    st.markdown(f"""
    <div class="sidebar-brand">
        <img src="data:image/jpeg;base64,{LOGO_B64}" alt="WE Logo">
        <div class="sidebar-brand-text">
            <strong>WE</strong>
            المصرية للاتصالات
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    st.markdown('<div class="sidebar-section-title">⚙️ Controls</div>', unsafe_allow_html=True)

    if st.button("🗑️  مسح المحادثة  /  Clear Chat"):
        st.session_state.messages = []
        st.session_state.assistant.history = []
        st.session_state.is_generating = False
        if "current_rag_prompt" in st.session_state:
            del st.session_state.current_rag_prompt
        st.rerun()

    # Features
    st.markdown('<div class="sidebar-section-title">⭐ Features</div>', unsafe_allow_html=True)

    features = [
        ("🔡", "Arabic / English Support"),
        ("🗣️", "Egyptian Dialect Support"),
        ("📚", "RAG-based Retrieval"),
        ("🧠", "Context-Aware Conversations"),
        ("🎧", "WE FAQ & Support Integration"),
    ]

    for icon, label in features:
        st.markdown(f"""
        <div class="feature-item">
            <div class="feature-icon">{icon}</div>
            <span class="feature-label">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    # Support Card
    st.markdown("""
    <div class="support-card">
        <div class="label">🎧 Support</div>
        <div class="title">Need more help?</div>
        <div class="subtitle">Call us at</div>
        <div class="number">📞 111</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="we-header">
    <div class="we-header-left">
        <img class="we-header-logo" src="data:image/jpeg;base64,{LOGO_B64}" alt="WE">
        <div>
            <h1 class="we-header-title">WE Intelligent Assistant</h1>
            <p class="we-header-subtitle">مساعدك الذكي لكل خدمات WE</p>
        </div>
    </div>
    <div class="we-header-badge">AI Powered</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. CHAT AREA
# ─────────────────────────────────────────────
if not st.session_state.messages:
    # Welcome Screen
    st.markdown(f"""
    <div class="welcome-container">
        <img class="welcome-icon" src="data:image/jpeg;base64,{LOGO_B64}" alt="WE">
        <h2 class="welcome-title">اهلاً! إزاي أقدر أساعدك؟</h2>
        <p class="welcome-sub">أنا مساعدك الذكي من WE — اسألني عن أي خدمة</p>
        <div class="welcome-chips">
            <span class="chip">باقات الموبايل</span>
            <span class="chip">باقات الإنترنت</span>
            <span class="chip">الخط الارضي</span>
            <span class="chip">خدمة العملاء</span>
            <span class="chip">العروض</span>
            <span class="chip">عن وي</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🟣"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(escape_markdown(msg["content"]))

    # GENERATE RESPONSE AFTER RERUN
    if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and st.session_state.is_generating
    ):
        # يقرأ الـ prompt المدمج بالملف إن وجد، وإلا يقرأ نص الرسالة العادية
        user_input_text = st.session_state.get("current_rag_prompt", st.session_state.messages[-1]["content"])

        with st.chat_message("assistant", avatar="🟣"):
            message_placeholder = st.empty()
            with st.spinner("جاري المعالجة..."):
                answer = st.session_state.assistant.chat(user_input_text)

            full_response = ""
            words = answer.split(" ")
            for chunk in words:
                full_response += chunk + " "
                message_placeholder.markdown(escape_markdown(full_response) + "▌")
                time.sleep(0.01)

            message_placeholder.markdown(escape_markdown(full_response))

        # SAVE RESPONSE
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        st.session_state.is_generating = False
        st.rerun()

# ─────────────────────────────────────────────
# 8. CHAT INPUT & PROCESSING
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 8. CHAT INPUT & PROCESSING
# ─────────────────────────────────────────────

# uploader reset state
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

# ───────────────── FILE & IMAGE UPLOAD + CLEAR CHAT ROW ─────────────────
# Adjusted column ratios to fit the new dynamic send button
action_col1, action_col2, action_col3, action_col4 = st.columns([2, 2, 1, 2])

with action_col1:
    uploaded_file = st.file_uploader(
        "📄 Upload File",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

with action_col2:
    uploaded_image = st.file_uploader(
        "🖼️ Upload Image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key=f"image_uploader_{st.session_state.image_uploader_key}"
    )

with action_col3:
    if st.button("🗑️ Clear", help="Clear Chat"):
        st.session_state.messages = []
        st.session_state.assistant.history = []
        st.session_state.is_generating = False

        if "current_rag_prompt" in st.session_state:
            del st.session_state.current_rag_prompt

        st.session_state.file_uploader_key += 1
        st.session_state.image_uploader_key += 1
        st.rerun()

with action_col4:
    send_file_btn = False
    # Dynamic Button: Only appears when a file is ready to be sent
    if uploaded_file or uploaded_image:
        send_file_btn = st.button("📤 إرسال المرفق", help="Send uploaded file without text")

# ───────────────── CHAT INPUT ─────────────────
user_input = st.chat_input(
    "اكتب رسالتك هنا...",
    disabled=st.session_state.is_generating
)

# ───────────────── PROCESSING ─────────────────
# Trigger if user types a message OR clicks the send attachment button
if (user_input or send_file_btn) and not st.session_state.is_generating:

    st.session_state.is_generating = True

    # Ensure we handle empty inputs gracefully if only a file was sent
    actual_input = user_input if user_input else ""

    rag_prompt = actual_input
    ui_content = actual_input

    extracted_text = ""
    attached_name = ""

    # ───────────────── PROCESS FILES ─────────────────
    if uploaded_file is not None:
        attached_name = uploaded_file.name
        with st.spinner("جاري استخراج النص من الملف..."):
            extracted_text = extract_text_from_file(uploaded_file)

    # ───────────────── PROCESS IMAGES OCR ─────────────────
    elif uploaded_image is not None:
        attached_name = uploaded_image.name
        with st.spinner("جاري استخراج النص من الصورة..."):
            extracted_text = extract_text_from_image(uploaded_image)

    # ───────────────── BUILD RAG PROMPT & FALLBACK LOGIC ─────────────────
    if uploaded_file or uploaded_image:
        if extracted_text and extracted_text.strip():
            # Standard path: Text found
            prefix = f"{actual_input}\n\n" if actual_input else ""

            rag_prompt = (
                f"{prefix}"
                f"📎 [محتوى الملف/الصورة {attached_name}]:\n"
                f"\"\"\"\n{extracted_text}\n\"\"\""
            )

            ui_content = (
                f"📎 **المرفق:** `{attached_name}`\n\n"
                f"{actual_input}"
            ).strip()
        else:
            # Fallback path: File uploaded but no text extracted
            ui_content = (
                f"📎 **المرفق:** `{attached_name}`\n\n"
                f"{actual_input}"
            ).strip()

            # Prevent the LLM from receiving a completely empty prompt
            if not actual_input:
                rag_prompt = f"المستخدم قام برفع ملف باسم '{attached_name}' ولكنه لا يحتوي على نص مقروء ولم يكتب رسالة."

    # save prompt
    st.session_state.current_rag_prompt = rag_prompt

    # add user msg
    st.session_state.messages.append({
        "role": "user",
        "content": ui_content
    })

    # reset uploaders after sending
    st.session_state.file_uploader_key += 1
    st.session_state.image_uploader_key += 1

    st.rerun()