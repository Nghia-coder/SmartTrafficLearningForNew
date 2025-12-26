import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
import random
import sqlite3
import datetime
from collections import defaultdict
import google.generativeai as genai

# =====================
# FIX LOG / ENV
# =====================
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

# =====================
# CONFIG
# =====================
# !!! THAY THẾ API KEY CỦA BẠN VÀO ĐÂY !!!
GEMINI_API_KEY = "" # Replace with your actual API key

MODEL_PATH = "last85.pt"
SIGNS_INFO_PATH = "signs_info.json"
SIGNS_DIR = "signs"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
DB_PATH = "users.db"
CACHE_DB = "gemini_cache.db"

st.set_page_config(page_title="Học Biển Báo Giao Thông", layout="wide")

# =====================
# CSS GIAO DIỆN
# =====================
# =====================
# CSS GIAO DIỆN
# =====================
st.markdown("""
    <style>
    * { box-sizing: border-box; }
    body { background-color: #fdfdfd; font-family: "Segoe UI", sans-serif; color: #333; }
    .header {
        background: linear-gradient(90deg, #457b9d, #1d3557);
        color: white; padding: 15px 30px; border-radius: 0 0 12px 12px;
        display: flex; align-items: center; gap: 15px;
    }
    .header img { height: 45px; }
    .header h1 { font-size: 1.6rem; margin: 0; font-weight: 600; }
    
    /* (SỬA LỚN) ÁP DỤNG MÀU XANH DƯƠNG CHO TẤT CẢ CÁC NÚT MẶC ĐỊNH (PRIMARY) */
    .stButton>button {
        /* Thay thế gradient cam/vàng bằng gradient xanh dương */
        background: linear-gradient(135deg, #457b9d, #1d3557); 
        color: white; border-radius: 8px; padding: 8px 20px;
        font-weight: 600; border: none; transition: all 0.2s ease;
    }
    .stButton>button:hover {
        /* Đảo lại gradient xanh dương khi hover */
        background: linear-gradient(135deg, #1d3557, #457b9d); 
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* (SỬA) Thêm :disabled style cho nút Quiz */
    .stButton>button:disabled {
        background: #ced4da;
        color: #6c757d;
        opacity: 0.7;
        cursor: not-allowed;
    }
    .stButton>button:disabled:hover {
        transform: none;
        box-shadow: none;
        background: #ced4da;
    }

    .sign-card {
        background-color: white; padding: 15px; border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1); margin-bottom: 15px;
        color: #333;
    }
    .footer {
        text-align: center; color: #6c757d; padding: 10px; font-size: 0.9rem;
    }

    /* --- CSS CHO TRANG CHAT --- */

    /* Cột 1 - Nút session */
    .st-emotion-cache-1jicfl2 .stButton>button {
        background: #f0f2f6;
        color: #333;
        font-weight: 500;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        text-align: left;
        padding: 12px 15px;
        transition: all 0.2s ease;
        width: 100%;
        margin-bottom: 5px;
    }
    .st-emotion-cache-1jicfl2 .stButton>button:hover {
        background: #e6e8eb;
        color: #111;
        transform: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-color: #d0d0d0;
    }
    /* Style cho nút session được chọn (dùng JS để thêm class/style) */
    .st-emotion-cache-1jicfl2 .stButton>button.selected-session {
         background: #e7f5ff;
         color: #1d3557;
         border-color: #457b9d;
    }


    /* (CẢI TIẾN) CĂN CHỈNH CONTAINER CHỨA NÚT CHAT SESSION VÀ NÚT XÓA */
    /* Target tất cả các HorizontalBlock (chứa nút session và nút xóa) */
    .st-emotion-cache-1jicfl2 [data-testid="stHorizontalBlock"] {
        align-items: center !important; /* Dùng !important để đảm bảo override và căn giữa dọc */
        margin-bottom: 5px;
    }

    /* (CẢI TIẾN) Nút xóa: Thiết lập kích thước cố định và căn giữa nội dung */
    /* Nút xóa */
    .st-emotion-cache-1jicfl2 .stButton>button[key*="delete_"] {
        background: #fdfdfd;
        color: #888;
        border: 1px solid #e0e0e0;
        /* (SỬA) Kích thước cố định để nút thành hình vuông */
        width: 35px; 
        height: 35px;
        line-height: 0; /* Loại bỏ line-height để căn giữa icon */
        padding: 0; /* Bỏ padding */
        display: flex; /* Dùng flex để căn icon */
        align-items: center; /* Căn giữa dọc */
        justify-content: center; /* Căn giữa ngang */
        flex-shrink: 0; 
        border-radius: 8px; /* Bo tròn góc */
    }
    .st-emotion-cache-1jicfl2 .stButton>button[key*="delete_"]:hover {
        background: #fee;
        color: #d00;
        border-color: #fbb;
        transform: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Cột 2 - Wrapper chính */
    .chat-main-wrapper { }

    /* Cột 2 - Ô đổi tên */
    .chat-main-wrapper [data-testid="stTextInput"] input {
        font-size: 1.25rem; font-weight: 600; color: #1d3557;
        border: none; border-bottom: 2px solid #e0e0e0; border-radius: 0; padding: 8px 0px;
    }
    .chat-main-wrapper [data-testid="stTextInput"] input:focus {
        box-shadow: none; border-bottom: 2px solid #457b9d;
    }

    /* Cột 2 - Khung chứa chat */
    .chat-main-wrapper [data-testid="stReportViewBlockContainer"] {
        border: 1px solid #e0e0e0; border-radius: 12px; padding: 10px; background-color: #ffffff;
    }

    /* Chat Bubbles */
    .chat-main-wrapper [data-testid="chatMessageContent"] {
        padding: 12px 18px; border-radius: 20px; max-width: 85%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-main-wrapper div[data-testid="stChatMessage"]:has([data-testid="stChatAvatarIcon-assistant"]) [data-testid="chatMessageContent"] {
        background-color: #f0f2f6; color: #333;
    }
    .chat-main-wrapper div[data-testid="stChatMessage"]:has([data-testid="stChatAvatarIcon-user"]) [data-testid="chatMessageContent"] {
        background-color: #457b9d; color: white;
    }
    .chat-main-wrapper div[data-testid="stChatMessage"]:has([data-testid="stChatAvatarIcon-user"]) {
        display: flex; flex-direction: row-reverse;
    }
    .chat-main-wrapper div[data-testid="stChatMessage"]:has([data-testid="stChatAvatarIcon-user"]) [data-testid="chatMessageContent"] * {
        text-align: left; color: white;
    }

    /* Avatars */
    .chat-main-wrapper [data-testid="stChatAvatarIcon-user"] div,
    .chat-main-wrapper [data-testid="stChatAvatarIcon-assistant"] div {
        background-color: #e0e0e0; padding: 4px;
    }
    .chat-main-wrapper [data-testid="stChatAvatarIcon-user"] div { background-color: #fb8500; }
    .chat-main-wrapper [data-testid="stChatAvatarIcon-user"] svg { fill: white; }
    .chat-main-wrapper [data-testid="stChatAvatarIcon-assistant"] div { background-color: #1d3557; }
    .chat-main-wrapper [data-testid="stChatAvatarIcon-assistant"] svg { fill: white; }

    /* Placeholder */
    .chat-placeholder {
        display: flex; align-items: center; justify-content: center; height: 500px;
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px;
    }
    .chat-placeholder h3 { color: #888; font-weight: 400; text-align: center; padding: 20px; }

    /* --- HẾT CSS CHO TRANG CHAT --- */
            
    /* Áp dụng cho TẤT CẢ các nút trong sidebar */
    [data-testid="stSidebar"] .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
        margin-bottom: 5px; 
        border: 1px solid transparent; 
        text-align: left; 
        padding-left: 15px;
    }

    /* Nút không được chọn (secondary) */
    [data-testid="stSidebar"] .stButton>button[data-kind="secondary"] {
        background-color: #f0f8ff; 
        color: #1d3557; 
        border-color: #a8dadc; 
    }

    /* Hover nút không được chọn */
    [data-testid="stSidebar"] .stButton>button[data-kind="secondary"]:hover {
        background-color: #e0f7fa; 
        color: #1d3557;
        border-color: #457b9d; 
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(69, 123, 157, 0.2); 
    }

    /* Nút được chọn (primary) */
    [data-testid="stSidebar"] .stButton>button[data-kind="primary"] {
        background: linear-gradient(90deg, #457b9d, #1d3557);
        color: white;
        border-color: #1d3557;
        box-shadow: 0 4px 12px rgba(29, 53, 87, 0.3);
    }
    
    /* Hover nút được chọn (chỉ sáng lên một chút) */
    [data-testid="stSidebar"] .stButton>button[data-kind="primary"]:hover {
         filter: brightness(110%);
    }

    /* Bỏ viền focus mặc định (đã có style riêng) */
    [data-testid="stSidebar"] .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(69, 123, 157, 0.5); 
    }
    /* --- HẾT CSS NÚT SIDEBAR --- */
    </style>
""", unsafe_allow_html=True)
# =====================
# HEADER
# =====================
st.markdown("""
<div class="header">
    <img src="https://cdn-icons-png.flaticon.com/512/3097/3097144.png">
    <h1>Học Biển Báo Giao Thông 🚦</h1>
</div>
""", unsafe_allow_html=True)

# =====================
# INIT DATABASES
# =====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS quiz_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        score INTEGER NOT NULL,
        total_questions INTEGER NOT NULL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        session_id INTEGER DEFAULT 0 NOT NULL /* Add session_id with default */
    )''')

    # Ensure session_id column exists, add if not (handles older DB)
    c.execute("PRAGMA table_info(chat_history)")
    columns = [col[1] for col in c.fetchall()]
    if 'session_id' not in columns:
        c.execute("ALTER TABLE chat_history ADD COLUMN session_id INTEGER DEFAULT 0 NOT NULL")

    conn.commit()
    conn.close()

def init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gemini_cache (
        sign_name TEXT PRIMARY KEY,
        explanation TEXT
    )''')
    conn.commit()
    conn.close()

init_db()
init_cache_db()

# =====================
# USER FUNCTIONS
# =====================
def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE AND password = ?", (username, password))
    u = c.fetchone()
    conn.close()
    return u[0] if u else None

def save_quiz_history(uid, score, total):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO quiz_history (user_id, date, score, total_questions) VALUES (?, ?, ?, ?)",
              (uid, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), score, total))
    conn.commit()
    conn.close()

def get_history(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, score, total_questions FROM quiz_history WHERE user_id = ? ORDER BY date DESC", (uid,))
    h = c.fetchall()
    conn.close()
    return h

def get_learning_streak(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date FROM quiz_history WHERE user_id = ? ORDER BY date DESC", (uid,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return 0

    # Chuyển tất cả thành set ngày (YYYY-MM-DD)
    dates = [datetime.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S").date() for r in rows]
    unique_days = sorted(list(set(dates)), reverse=True)

    streak = 0
    today = datetime.date.today()

    for i, d in enumerate(unique_days):
        if i == 0:
            # Ngày đầu tiên phải là hôm nay hoặc hôm qua mới bắt đầu tính streak
            if d == today:
                streak = 1
            elif d == today - datetime.timedelta(days=1):
                streak = 1
                today = d  # bắt đầu đếm từ hôm qua
            else:
                break
        else:
            if d == today - datetime.timedelta(days=1):
                streak += 1
                today = d
            else:
                break

    return streak

# =====================
# CHAT HISTORY & SESSIONS
# =====================
def save_chat(uid, session_id, role, msg):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_id, session_id, role, message) VALUES (?, ?, ?, ?)", (uid, session_id, role, msg))
    conn.commit()
    conn.close()

def load_chat(uid, session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, message FROM chat_history WHERE user_id = ? AND session_id = ? ORDER BY id ASC", (uid, session_id))
    chats = c.fetchall()
    conn.close()
    return chats

def get_chat_sessions(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Migrate old chats (session_id = 0)
    c.execute("SELECT id FROM chat_history WHERE user_id = ? AND session_id = 0 LIMIT 1", (uid,))
    if c.fetchone():
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_sessions (user_id, session_name, created_at) VALUES (?, ?, ?)", (uid, "Cuộc trò chuyện cũ", date))
        new_session_id = c.lastrowid
        c.execute("UPDATE chat_history SET session_id = ? WHERE user_id = ? AND session_id = 0", (new_session_id, uid))
        conn.commit()

    c.execute("SELECT session_id, session_name FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC", (uid,))
    sessions = c.fetchall()
    conn.close()
    return sessions

def create_new_session(uid, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO chat_sessions (user_id, session_name, created_at) VALUES (?, ?, ?)", (uid, name, date))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id

def rename_session(session_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE chat_sessions SET session_name = ? WHERE session_id = ?", (new_name, session_id))
    conn.commit()
    conn.close()

def delete_chat_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# =====================
# YOLO DETECTION
# =====================
@st.cache_resource(show_spinner=False)
def load_model(p): return YOLO(p)

def load_signs(p): return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else {}

def detect(model, img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    r = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
    dets = []
    for box in r.boxes:
        cls_id = int(box.cls.cpu().numpy())
        name = model.model.names.get(cls_id, str(cls_id))
        dets.append({"class_name": name, "conf": float(box.conf.cpu().numpy()), "box": box.xyxy.cpu().numpy().tolist()[0]})
    return dets

def draw_boxes(img_pil, dets):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(img, d["class_name"], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# =====================
# AI GIAO THÔNG FUNCTIONS (CACHE & QUIZ)
# =====================
def gemini_explain(signs):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "...YOUR_API_KEY_HERE..." or not signs:
        return None

    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    results = {}

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        conn.close()
        return {sign: f"Lỗi cấu hình API của AI Giao Thông: {e}" for sign in signs}

    for sign in signs:
        c.execute("SELECT explanation FROM gemini_cache WHERE sign_name=?", (sign,))
        row = c.fetchone()
        if row:
            results[sign] = row[0]
        else:
            try:
                prompt = f"Hãy giải thích ngắn gọn và dễ hiểu bằng tiếng Việt về biển báo '{sign}'. Trình bày theo cú pháp:\nTên biển báo\nTác dụng\nMức phạt (theo luật giao thông Việt Nam năm 2025)."
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                results[sign] = text
                c.execute("INSERT OR REPLACE INTO gemini_cache (sign_name, explanation) VALUES (?, ?)", (sign, text))
                conn.commit()
            except Exception as e:
                results[sign] = f"Lỗi khi gọi API của AI Giao Thông: {e}"
    conn.close()
    return results

# (MỚI) Hàm giải thích câu hỏi Quiz
def get_ai_explanation(question, user_answer, correct_answer):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "...YOUR_API_KEY_HERE...":
        return "Vui lòng cấu hình API Key để nhận giải thích chi tiết."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            f"Giải thích ngắn gọn bằng tiếng Việt tại sao câu trả lời cho câu hỏi sau:\n"
            f"Câu hỏi: \"{question}\"\n"
            f"Là \"{correct_answer}\" chứ không phải \"{user_answer}\".\n"
            "Chỉ tập trung vào luật và ý nghĩa của biển báo. Không cần chào hỏi."
        )

        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Lỗi khi gọi AI Giao Thông: {e}"


# =====================
# SESSION
# =====================
# Initialize session state variables if they don't exist
# (Mới)
# =====================
# SESSION (ĐÃ SỬA)
# =====================
# Initialize session state variables if they don't exist
# (Mới)
default_states = {
    "user_id": None,
    "username": None,
    "current_session_id": None,
    "session_to_delete": None,
    "quiz_data": [],
    "q": 0,
    "score": 0,
    "answered": False,
    "ai_explanation": None,
    "is_loading_explanation": False,
    "page": "🏷️ Nhận diện",
    # (MỚI) Thêm state cho tab đăng nhập/đăng ký
    "login_tab": "🔐 Đăng nhập" 
}
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =====================
# LOGIN / REGISTER (ĐÃ SỬA LẠI ĐỂ DÙNG BUTTON VÀ STYLE)
# =====================
if not st.session_state.user_id:
    st.markdown("<h2 style='text-align:center;margin-top:20px;'>Học Biển Báo Giao Thông 🚗</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;margin-bottom:20px;'>Chọn hành động:</p>", unsafe_allow_html=True)

    # Thay st.radio bằng st.button trong columns
    tab_col1, tab_col2 = st.columns(2)
    
    # --- Nút Đăng nhập ---
    with tab_col1:
        is_login_tab = (st.session_state.login_tab == "🔐 Đăng nhập")
        # Đặt type thành "primary" để có màu xanh dương nếu được chọn
        btn_type = "primary" if is_login_tab else "secondary" 
        
        # Thêm key để nút có thể sử dụng CSS đã tùy chỉnh (bắt chước nút sidebar)
        if st.button("🔐 Đăng nhập", use_container_width=True, type=btn_type, key="login_tab_btn"):
            st.session_state.login_tab = "🔐 Đăng nhập"
            st.rerun()

    # --- Nút Đăng ký ---
    with tab_col2:
        is_register_tab = (st.session_state.login_tab == "📝 Đăng ký")
        # Đặt type thành "primary" để có màu xanh dương nếu được chọn
        btn_type = "primary" if is_register_tab else "secondary"
        
        # Thêm key để nút có thể sử dụng CSS đã tùy chỉnh
        if st.button("📝 Đăng ký", use_container_width=True, type=btn_type, key="register_tab_btn"):
            st.session_state.login_tab = "📝 Đăng ký"
            st.rerun()

    st.markdown("---") # Đường kẻ phân cách

    # Hiển thị form dựa trên trạng thái đã chọn
    if st.session_state.login_tab == "🔐 Đăng nhập":
        st.subheader("Đăng nhập")
        with st.form(key="login_form"):
            u = st.text_input("Tên đăng nhập")
            p = st.text_input("Mật khẩu", type="password")
            # Nút submit này sẽ dùng CSS gradient cam/vàng mặc định của stButton, 
            # không bị ảnh hưởng bởi CSS sidebar.
            login_button = st.form_submit_button("Đăng nhập", use_container_width=True) 

        if login_button:
            uid = login_user(u, p)
            if uid:
                st.session_state.user_id = uid
                st.session_state.username = u
                st.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")

    else: # Tab Đăng ký
        st.subheader("Đăng ký tài khoản mới")
        with st.form(key="register_form"):
            u = st.text_input("Tên đăng nhập mới")
            p = st.text_input("Mật khẩu", type="password")
            cp = st.text_input("Xác nhận mật khẩu", type="password")
            # Nút submit này sẽ dùng CSS gradient cam/vàng mặc định của stButton.
            register_button = st.form_submit_button("Đăng ký", use_container_width=True) 

        if register_button:
            if p != cp:
                st.error("Mật khẩu không khớp.")
            elif not u or not p:
                 st.error("Tên đăng nhập và mật khẩu không được để trống.")
            elif register_user(u, p):
                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
            else:
                st.error("Tên đăng nhập đã tồn tại (không phân biệt hoa thường).")
    st.stop()

# =====================
# MAIN MENU
# =====================
st.sidebar.title(f"Xin chào, {st.session_state.username} 👋")
if st.sidebar.button("Đăng xuất"):
    # Reset all relevant session state keys on logout
    for key in default_states.keys():
        st.session_state[key] = default_states[key]
    st.rerun()

# (Mới)
st.sidebar.markdown("---")
st.sidebar.markdown("##### Chọn trang") # Thêm tiêu đề cho nhóm nút

# Danh sách các trang
pages_list = ["🏷️ Nhận diện", "📘 Tài liệu", "💬 Chat với AI Giao Thông", "🧩 Quiz", "📜 Lịch sử Quiz"]

# Đọc trang hiện tại từ session state
current_page = st.session_state.page

# Tạo các nút
for p in pages_list:
    # Đặt loại nút: "primary" (nổi bật) nếu là trang hiện tại, "secondary" (thường) nếu không
    btn_type = "primary" if (p == current_page) else "secondary"
    
    if st.sidebar.button(p, use_container_width=True, type=btn_type):
        st.session_state.page = p
        st.rerun() # Chạy lại script ngay lập tức để cập nhật giao diện

# Gán biến 'page' để phần còn lại của script hoạt động
page = st.session_state.page

# Load model and signs info (handle potential errors)
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        st.error(f"Lỗi: Không tìm thấy file model tại '{MODEL_PATH}'.")
        st.stop()

    if os.path.exists(SIGNS_INFO_PATH):
        signs_info = load_signs(SIGNS_INFO_PATH)
    else:
        st.warning(f"Cảnh báo: Không tìm thấy file '{SIGNS_INFO_PATH}'. Chức năng Tài liệu và Quiz sẽ bị ảnh hưởng.")
        signs_info = {}
except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu cần thiết: {e}")
    st.stop()


# =====================
# PAGE 1 - NHẬN DIỆN
# =====================
if page == "🏷️ Nhận diện":
    st.title("🚦 Nhận diện Biển Báo Giao Thông")
    source = st.radio("Nguồn ảnh", ["📸 Camera", "🖼️ Upload"])
    image_file = st.camera_input("Chụp ảnh") if source == "📸 Camera" else st.file_uploader("Tải ảnh", type=["jpg","jpeg","png"])

    if image_file and st.button("🔍 Nhận diện biển báo"):
        try:
            image = Image.open(image_file).convert("RGB")
            with st.spinner("Đang nhận diện..."):
                dets = detect(model, image)

            if not dets:
                st.warning("Không phát hiện biển báo nào.")
            else:
                st.success(f"Phát hiện {len(dets)} biển báo!")
                res_img = draw_boxes(image, dets)

                names = list(set(d["class_name"] for d in dets)) # Unique names
                col1, col2 = st.columns(2)
                with col1:
                    st.image(res_img, caption="Ảnh đã nhận diện", use_container_width=True)
                with col2:
                    if os.path.isdir(SIGNS_DIR):
                        for n in names:
                            path = os.path.join(SIGNS_DIR, f"{n}.png")
                            if os.path.exists(path):
                                st.image(path, caption=n, use_container_width='auto') # Use auto width
                            else :
                                st.caption(f"(Ảnh {n}.png bị thiếu)")
                    else:
                        st.caption(f"(Thư mục '{SIGNS_DIR}' không tồn tại)")


                if GEMINI_API_KEY and GEMINI_API_KEY != "...YOUR_API_KEY_HERE...":
                    with st.spinner("🔎 AI Giao Thông đang phân tích..."):
                        explains = gemini_explain(names)

                    if explains:
                        st.markdown("### 💬 Thông tin từ AI Giao Thông")
                        for name, text in explains.items():
                            st.markdown(f"**🛑 {name}:**\n\n{text}\n---")
                    else:
                        st.error("Không thể lấy dữ liệu từ AI Giao Thông. Key API có thể bị sai hoặc có lỗi xảy ra.")
                else:
                    st.info("💡 Vui lòng cấu hình GEMINI_API_KEY trong code để xem thông tin AI phân tích.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")


# =====================
# PAGE 2 - TÀI LIỆU
# =====================
elif page == "📘 Tài liệu":
    st.title("📘 Tài liệu Học Biển Báo")
    if not signs_info:
        st.warning("Chưa có dữ liệu biển báo (thiếu file 'signs_info.json').")
    else:
        grouped = defaultdict(list)
        for k, info in signs_info.items():
            grouped[info.get("category","Khác")].append((k, info))

        search_term = st.text_input("Tìm kiếm biển báo (theo tên hoặc mô tả):")

        for cat, items in grouped.items():
            # Filter items based on search term
            filtered_items = [
                (k, info) for k, info in items
                if not search_term or \
                   search_term.lower() in info.get('title', k).lower() or \
                   search_term.lower() in info.get('description', '-').lower()
            ]

            if filtered_items: # Only show expander if there are matching items
                with st.expander(f"📗 {cat} ({len(filtered_items)})"):
                    cols = st.columns(3)
                    for i, (k, info) in enumerate(filtered_items):
                        with cols[i % 3]:
                            st.markdown(f"<div class='sign-card'>", unsafe_allow_html=True)
                            if os.path.isdir(SIGNS_DIR):
                                path = os.path.join(SIGNS_DIR, f"{k}.png")
                                if os.path.exists(path):
                                    st.image(path, width=120)
                                else:
                                    st.caption(f"(Ảnh {k}.png bị thiếu)")
                            else:
                                st.caption(f"(Thư mục '{SIGNS_DIR}' không tồn tại)")

                            st.markdown(f"**{info.get('title',k)}**")
                            st.write(info.get("description","-"))
                            st.markdown("</div>", unsafe_allow_html=True)
        # Indicate if no results found
        if search_term and not any(filtered_items for cat, items in grouped.items() for filtered_items in [[(k, info) for k, info in items if search_term.lower() in info.get('title', k).lower() or search_term.lower() in info.get('description', '-').lower()]]):
             st.info(f"Không tìm thấy biển báo nào khớp với '{search_term}'.")


# =====================
# PAGE 3 - CHAT VỚI AI GIAO THÔNG
# =====================
elif page == "💬 Chat với AI Giao Thông":
    st.title("💬 Trò chuyện với AI Giao Thông")

    if not GEMINI_API_KEY or GEMINI_API_KEY == "...YOUR_API_KEY_HERE...":
        st.info("🔑 Vui lòng cấu hình GEMINI_API_KEY trong code để sử dụng chat.")
    else:
        col1, col2 = st.columns([1, 3])
        chat_sessions = get_chat_sessions(st.session_state.user_id)

        # --- CỘT 1: LỊCH SỬ CHAT VÀ QUẢN LÝ ---
        with col1:
            st.markdown("### Cuộc trò chuyện")
            if st.button("💬 Tạo cuộc trò chuyện mới", use_container_width=True):
                new_id = create_new_session(st.session_state.user_id, f"Cuộc trò chuyện {len(chat_sessions) + 1}")
                st.session_state.current_session_id = new_id
                st.session_state.session_to_delete = None
                st.rerun()
            st.markdown("---")

            for session_id, session_name in chat_sessions:
                s_col1, s_col2 = st.columns([5, 1])
                with s_col1:
                    is_selected = (session_id == st.session_state.current_session_id and st.session_state.session_to_delete is None)
                    button_key = f"session_{session_id}"
                    if st.button(session_name, key=button_key, use_container_width=True):
                         st.session_state.current_session_id = session_id
                         st.session_state.session_to_delete = None
                         st.rerun()
                    # Apply selected style using JS
                    if is_selected:
                        st.markdown(f"""
                            <script>
                                var buttons = window.parent.document.querySelectorAll('button');
                                buttons.forEach(function(button) {{
                                    if (button.innerText.trim() === "{session_name}") {{
                                        button.classList.add('selected-session'); // Add class for styling
                                    }} else {{
                                        button.classList.remove('selected-session'); // Remove class if not selected
                                    }}
                                }});
                            </script>
                         """, unsafe_allow_html=True)

                with s_col2:
                    if st.button("🗑️", key=f"delete_{session_id}", use_container_width=True):
                        st.session_state.session_to_delete = session_id
                        if session_id == st.session_state.current_session_id:
                             st.session_state.current_session_id = None
                        st.rerun()

        # --- CỘT 2: KHUNG CHAT CHÍNH ---
        with col2:
            st.markdown('<div class="chat-main-wrapper">', unsafe_allow_html=True)
            if st.session_state.session_to_delete is not None:
                # --- TRẠNG THÁI 1: XÁC NHẬN XÓA ---
                st.markdown('<div class="chat-placeholder"><h3>Bạn có chắc muốn xóa lịch sử trò chuyện này không?</h3></div>', unsafe_allow_html=True)
                session_name_to_delete = next((name for sid, name in chat_sessions if sid == st.session_state.session_to_delete), "Không rõ")
                st.warning(f"Đang xóa: **{session_name_to_delete}**")
                del_col1, del_col2 = st.columns(2)
                with del_col1:
                    if st.button("💔 Có, tôi muốn", use_container_width=True, key="delete_confirm"):
                        delete_chat_session(st.session_state.session_to_delete)
                        st.session_state.session_to_delete = None
                        # current_session_id already set to None above if needed
                        st.rerun()
                with del_col2:
                    if st.button("💖 Không, kí ức này quá đẹp", use_container_width=True, key="delete_cancel"):
                        # If cancel was clicked, restore current session id if it was unset
                        if st.session_state.current_session_id is None and st.session_state.session_to_delete is not None:
                            st.session_state.current_session_id = st.session_state.session_to_delete
                        st.session_state.session_to_delete = None
                        st.rerun()

            elif st.session_state.current_session_id is None:
                # --- TRẠNG THÁI 2: PLACEHOLDER ---
                st.markdown('<div class="chat-placeholder"><h3>Chọn một cuộc trò chuyện hoặc tạo cuộc trò chuyện mới để bắt đầu.</h3></div>', unsafe_allow_html=True)
            else:
                # --- TRẠNG THÁI 3: HIỂN THỊ CHAT ---
                current_session_id = st.session_state.current_session_id
                current_session_name = next((name for sid, name in chat_sessions if sid == current_session_id), "...")
                new_name = st.text_input("Tên cuộc trò chuyện:", value=current_session_name, key=f"rename_{current_session_id}", label_visibility="collapsed")
                if new_name != current_session_name and new_name:
                    rename_session(current_session_id, new_name)
                    st.rerun()

                chats = load_chat(st.session_state.user_id, current_session_id)
                chat_container = st.container(height=450)
                with chat_container:
                    for role, msg in chats:
                        with st.chat_message(role):
                            st.markdown(msg)

                user_msg = st.chat_input("Nhập câu hỏi về biển báo giao thông...")
                if user_msg:
                    save_chat(st.session_state.user_id, current_session_id, "user", user_msg)
                    # Display user message immediately before calling API
                    with chat_container:
                         with st.chat_message("user"):
                              st.markdown(user_msg)
                    # Call API and display response
                    try:
                        genai.configure(api_key=GEMINI_API_KEY)
                        # (SỬA) Đảm bảo có system instruction hợp lệ
                        system_instruction = (
                            "Bạn là chuyên gia về luật giao thông Việt Nam. "
                            "Trả lời ngắn gọn, dễ hiểu, chỉ tập trung vào biển báo và mức phạt liên quan. "
                            "Hãy trả lời các câu hỏi của người dùng dựa trên lịch sử trò chuyện trước đó CỦA CUỘC TRÒ CHUYỆN NÀY."
                        )
                        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system_instruction)
                        # Lấy lại chats *sau khi* lưu tin nhắn user để đưa vào history
                        chats_with_new_user_msg = load_chat(st.session_state.user_id, current_session_id)
                        api_history_formatted = [{"role": ("model" if db_role == "assistant" else "user"), "parts": [{"text": msg}]} for db_role, msg in chats_with_new_user_msg[:-1]] # Bỏ tin nhắn user cuối cùng khỏi history API

                        chat_session = model.start_chat(history=api_history_formatted)
                        resp = chat_session.send_message(user_msg) # Send only the new message
                        reply = resp.text.strip()
                        save_chat(st.session_state.user_id, current_session_id, "assistant", reply)
                    except Exception as e:
                        reply = f"Lỗi khi gọi API của AI Giao Thông: {e}"
                        # Save error message as assistant response? Optional.
                        save_chat(st.session_state.user_id, current_session_id, "assistant", reply) # Lưu lỗi để hiển thị
                        st.error(reply) # Show error inline
                    # Rerun to display the new assistant message from db
                    st.rerun() # Rerun AFTER saving assistant message


            st.markdown('</div>', unsafe_allow_html=True)


# =====================
# (NÂNG CẤP) PAGE 4 - QUIZ
# =====================
elif page == "🧩 Quiz":
    st.title("🧩 Kiểm Tra Kiến Thức Biển Báo")
    if not signs_info:
        st.warning("Chưa có dữ liệu biển báo (thiếu file 'signs_info.json').")
    else:
        # --- Khởi tạo Quiz ---
        # Initialize only if 'quiz_data' is missing or empty
        if not st.session_state.quiz_data:
            random.seed()
            items = list(signs_info.items())
            qs = []
            num_questions = min(10, len(items))
            valid_items = [(k, info) for k, info in items if info.get("description")]

            if len(valid_items) < num_questions or len(valid_items) < 4: # Need at least 4 items for 3 wrong answers
                st.warning("Không đủ câu hỏi hợp lệ trong dữ liệu để tạo quiz.")
                if len(valid_items) < 4:
                    st.stop()
                num_questions = len(valid_items)


            sampled_items = random.sample(valid_items, num_questions)
            all_descriptions = [info.get("description", "") for _, info in valid_items]

            for k, info in sampled_items:
                correct = info.get("description","")
                possible_wrongs = list(set(all_descriptions) - {correct})
                if len(possible_wrongs) < 3: continue

                wrongs = random.sample(possible_wrongs, 3)
                opts = wrongs + [correct]
                random.shuffle(opts)

                qs.append({
                    "q": f"Ý nghĩa của biển báo **{info.get('title',k)}** là gì?",
                    "sign_key": k,
                    "options": opts,
                    "answer_index": opts.index(correct),
                    "correct_answer_text": correct
                })

            st.session_state.quiz_data = qs[:num_questions]
            st.session_state.q = 0
            st.session_state.score = 0
            st.session_state.answered = False
            st.session_state.ai_explanation = None
            st.session_state.is_loading_explanation = False
            st.session_state.quiz_results_saved = False

        # --- Hiển thị câu hỏi hoặc kết quả ---
        if st.session_state.quiz_data and st.session_state.q < len(st.session_state.quiz_data):
            qs = st.session_state.quiz_data
            q_idx = st.session_state.q
            q_data = qs[q_idx]

            st.progress((q_idx + 1) / len(qs))
            st.subheader(q_data["q"])

            if os.path.isdir(SIGNS_DIR):
                path = os.path.join(SIGNS_DIR, f"{q_data['sign_key']}.png")
                if os.path.exists(path): st.image(path, width=150)
                else: st.caption(f"(Ảnh {q_data['sign_key']}.png bị thiếu)")
            else: st.caption(f"(Thư mục '{SIGNS_DIR}' không tồn tại)")

            # Store selected option using session state key specific to question index
            session_key_selected_option = f"selected_option_{q_idx}"
            if session_key_selected_option not in st.session_state:
                 st.session_state[session_key_selected_option] = q_data["options"][0] # Default

            def update_selection():
                # Update session state with the value from the radio button itself
                st.session_state[session_key_selected_option] = st.session_state[f"radio_{q_idx}"]

            # Find the index of the currently selected option in state
            try:
                current_selection_index = q_data["options"].index(st.session_state[session_key_selected_option])
            except ValueError:
                current_selection_index = 0 # Fallback if state is somehow invalid

            selected = st.radio(
                "Chọn đáp án:",
                q_data["options"],
                key=f"radio_{q_idx}", # Key for the radio widget
                index=current_selection_index,
                on_change=update_selection,
                disabled=st.session_state.answered
            )
            # Use the value from session state for logic
            current_selection = st.session_state[session_key_selected_option]

            # --- Hiển thị các nút ---
            col1, col2 = st.columns(2)
            with col1:
                answer_button_disabled = st.session_state.answered
                if st.button("Trả lời", use_container_width=True, disabled=answer_button_disabled, key=f"answer_btn_{q_idx}"):
                    st.session_state.answered = True
                    selected_index = q_data["options"].index(current_selection) # Check against the selection from state

                    if selected_index == q_data["answer_index"]:
                        st.success("✅ Chính xác!")
                        st.session_state.score += 1
                        st.session_state.is_loading_explanation = False
                    else:
                        st.error(f"❌ Sai! Đáp án đúng: {q_data['correct_answer_text']}")
                        st.session_state.is_loading_explanation = True

            # --- Logic gọi AI (chạy sau khi rerun từ nút Trả lời) ---
            if st.session_state.answered and st.session_state.is_loading_explanation and st.session_state.ai_explanation is None:
                with st.spinner("AI Giao Thông đang giải thích..."):
                    user_answer_text = current_selection # Use selection from state
                    explanation = get_ai_explanation(q_data["q"], user_answer_text, q_data["correct_answer_text"])
                    st.session_state.ai_explanation = explanation
                st.session_state.is_loading_explanation = False
                st.rerun() # Rerun AGAIN to display explanation and enable Next button

            with col2:
                next_button_disabled = not st.session_state.answered or st.session_state.is_loading_explanation
                if st.button("Câu tiếp", use_container_width=True, disabled=next_button_disabled, key=f"next_btn_{q_idx}"):
                    st.session_state.q += 1
                    # Reset states for the next question
                    st.session_state.answered = False
                    st.session_state.ai_explanation = None
                    st.session_state.is_loading_explanation = False
                    # No need to delete selected_option state, it will be ignored for the next q_idx
                    st.rerun()

            # --- Hiển thị giải thích AI ---
            if st.session_state.ai_explanation and not st.session_state.is_loading_explanation:
                st.info(st.session_state.ai_explanation)

        elif st.session_state.quiz_data:
             # --- Màn hình tổng kết ---
            score = st.session_state.score
            total = len(st.session_state.quiz_data)
            st.success(f"Hoàn thành! Bạn đã trả lời đúng {score}/{total} câu.")
            if not st.session_state.get("quiz_results_saved", False):
                save_quiz_history(st.session_state.user_id, score, total)
                st.session_state.quiz_results_saved = True # Đánh dấu đã lưu

            streak = get_learning_streak(st.session_state.user_id)
            proficiency = "Xuất Sắc"
            if total > 0:
                rate = score / total
                if rate <= 0.4: proficiency = "Yếu"
                elif rate <= 0.6: proficiency = "Trung Bình"
                elif rate <= 0.8: proficiency = "Khá"
            else: proficiency = "N/A"

            # Check if quiz_summary class exists before applying markdown
            # Assuming quiz-summary class is defined in CSS
            st.markdown('<div class="quiz-summary">', unsafe_allow_html=True)
            m_col1, m_col2 = st.columns(2)
            with m_col1: st.metric("Xếp loại", proficiency)
            with m_col2: st.metric("Chuỗi ngày học", f"{streak} ngày 🔥")
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Làm lại Quiz"):
                # Clear quiz-specific session state before rerunning
                keys_to_delete = ["quiz_data", "q", "score", "answered", "ai_explanation", "is_loading_explanation", "quiz_results_saved"]
                # Also remove any potentially lingering selection states
                # Need the previous total to clear selections properly
                prev_total = total if total > 0 else 10 # Estimate if total was 0
                for i in range(prev_total):
                    keys_to_delete.append(f"selected_option_{i}")

                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                # Re-initialize quiz_data to empty list to trigger re-creation
                st.session_state.quiz_data = []
                st.rerun()
        else:
             st.info("Bắt đầu làm quiz mới bằng cách tải lại trang hoặc điều hướng.")
             if st.button("Bắt đầu Quiz mới"):
                 keys_to_delete = ["quiz_data", "q", "score", "answered", "ai_explanation", "is_loading_explanation", "quiz_results_saved"]
                 for key in keys_to_delete:
                     if key in st.session_state:
                         del st.session_state[key]
                 # Re-initialize quiz_data to empty list to trigger re-creation
                 st.session_state.quiz_data = []
                 st.rerun()


# =====================
# PAGE 5 - LỊCH SỬ QUIZ
# =====================
elif page == "📜 Lịch sử Quiz":
    st.title("📜 Lịch Sử Làm Quiz")
    try:
        h = get_history(st.session_state.user_id)
        if not h:
            st.info("Chưa có lịch sử làm quiz.")
        else:
            st.markdown("Kết quả các lần kiểm tra gần nhất của bạn:")
            for d, s, t in h:
                try:
                    date_obj = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
                    formatted_date = date_obj.strftime("%H:%M %d/%m/%Y")
                except ValueError: formatted_date = d
                st.markdown(f"<div class='sign-card'><b>Ngày:</b> {formatted_date}<br><b>Điểm:</b> {s}/{t}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Không thể tải lịch sử quiz: {e}")


# =====================
# FOOTER
# =====================
st.markdown("<div class='footer'>Ứng dụng học biển báo giao thông – YOLOv8 + AI Giao Thông 🌤</div>", unsafe_allow_html=True)

