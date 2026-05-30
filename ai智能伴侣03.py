"""
AI智能伴侣 v3.0 — 一个基于 Streamlit + DeepSeek 的 AI 聊天应用
================================================================
你可以和自定义性格、姓名的 AI 伴侣聊天。
聊天记录会自动保存到本地 session/ 文件夹，支持多会话管理和历史回顾。

运行方式：在终端执行 `streamlit run ai智能伴侣03.py`
"""

# ============================================================
# 第一部分：导入依赖库
# ============================================================
# streamlit：用来快速搭建网页界面的框架
# os：操作文件和文件夹（创建目录、删除文件等）
# openai：调用大模型 API 的官方库（这里用来调 DeepSeek）
# datetime：获取当前日期时间，用来给会话文件命名
# json：把 Python 数据转成 JSON 格式存到文件，或者从文件读回来
# uuid：生成唯一的随机 ID，防止会话名重复
import streamlit as st
import os
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import json
import uuid

# ============================================================
# 第二部分：页面全局配置（必须放在最前面，且只能调用一次）
# ============================================================
st.set_page_config(
    page_title="AI智能伴侣",          # 浏览器标签页上的标题
    page_icon="🥰",                   # 标签页上的小图标
    layout="wide",                   # 页面布局：wide=宽屏模式
    initial_sidebar_state="expanded", # 侧边栏默认展开
    menu_items={}                    # 右上角菜单项（空=隐藏）
)

# ============================================================
# 第三部分：全局常量（不会变的配置值，统一放这里方便修改）
# ============================================================

# 会话文件存放的目录名
SESSION_DIR = "session"

# DeepSeek API 的地址（兼容 OpenAI 格式）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 使用的模型名称
MODEL_NAME = "deepseek-v4-pro"

# 默认的伴侣设定
DEFAULT_NICK_NAME = "香香"              # 默认姓名
DEFAULT_CHARACTER = "温柔可爱的河南妹"   # 默认性格描述

# Logo 图片的路径（优先用脚本同目录下的 sc/logo.png，兼容本地和云端）
# os.path.dirname(__file__) 获取当前脚本所在的文件夹
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(_SCRIPT_DIR, "sc", "logo.png")
# 如果上面路径不存在，回退到你的本地绝对路径
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = "C:/Users/石金/Desktop/PythonProject/.venv/sc/logo.png"

# ============================================================
# 第四部分：登录系统（保护你的 API 额度不被陌生人消耗）
# ============================================================
# 账号密码从环境变量读取
#   - 本地运行时：使用默认值 admin / 123456
#   - 云端部署时：在 Streamlit Secrets 里设置 APP_USERNAME 和 APP_PASSWORD
#   - 你也可以在终端里先执行 export 来修改：
#       export APP_USERNAME="你的账号"
#       export APP_PASSWORD="你的密码"
APP_USERNAME = os.getenv("APP_USERNAME", "admin")      # 登录账号
APP_PASSWORD = os.getenv("APP_PASSWORD", "123456")     # 登录密码

# 初始化登录状态（st.session_state 里的变量会在页面刷新后保留）
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # False=还没登录


def check_login(username, password):
    """验证账号和密码是否正确，返回 True 或 False"""
    return username == APP_USERNAME and password == APP_PASSWORD


# ---- 登录守卫：如果没登录，只显示登录框，后面的聊天界面全部隐藏 ----
if not st.session_state.logged_in:
    # 用三列布局把登录框推到页面中央（左:中:右 = 1:2:1）
    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    with col_center:
        st.title("🔐 AI智能伴侣")
        st.caption("请输入账号密码登录")

        # st.form 创建一个表单，点"登录"按钮时才一次性提交（不会每打一个字就刷新）
        with st.form("login_form"):
            username = st.text_input("账号", placeholder="请输入账号")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submitted = st.form_submit_button("登 录", use_container_width=True)

            if submitted:
                if check_login(username, password):
                    # 登录成功：标记已登录，刷新页面进入聊天界面
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ 账号或密码错误，请重试")

    # st.stop() 让 Streamlit 在此处停止，后面的聊天界面代码都不会执行
    st.stop()


# ============================================================
# 第五部分：系统提示词模板
# ============================================================
# %s 是占位符，后面会用真实的姓名和性格替换进去
# 这个提示词会告诉 AI："你现在要扮演某某角色，按以下规则聊天"
SYSTEM_PROMPT_TEMPLATE = (
    "你是%s，现在是用户的真实伴侣，请完全代入伴侣角色。\n"
    "规则：\n"
    "1.每次只回1条消息\n"
    "2.禁止任何场景或状态描述性文字\n"
    "3.匹配用户的语言\n"
    "4.回复简短，像微信聊天一样\n"
    "5.有需要的话可以用🥰等emoji表情\n"
    "6.用符合伴侣性格的方式对话\n"
    "7.回复的内容，要充分体现伴侣的性格特征\n"
    "伴侣性格：%s\n"
    "你必须严格遵守上述规则来回复用户"
)


# ============================================================
# 第六部分：工具函数（每个函数负责一个独立的小功能）
# ============================================================

def generate_session_name():
    """
    生成一个唯一的会话名称
    格式：年月日_时分秒_随机6位ID
    例如：2026-05-29_14-30-00_a1b2c3

    加随机ID是为了防止你在一秒内点两次"新建会话"导致文件名冲突
    """
    # strftime：把日期时间格式化成字符串
    time_str = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    # uuid.uuid4().hex[:6]：生成一个随机字符串，取前6位
    random_id = uuid.uuid4().hex[:6]
    return f"{time_str}_{random_id}"


def ensure_session_dir():
    """
    确保 session 文件夹存在
    如果不存在就创建一个，避免后面保存文件时报错"找不到目录"
    """
    if not os.path.exists(SESSION_DIR):
        os.mkdir(SESSION_DIR)


def get_session_file_path(session_name):
    """
    根据会话名称，返回对应的文件完整路径
    例如：session_name="2026-05-29_xxx" → "session/2026-05-29_xxx.json"

    这个函数只是拼接路径，不会真的创建或读取文件
    """
    return os.path.join(SESSION_DIR, f"{session_name}.json")


def save_session():
    """
    把当前会话的所有信息保存到 JSON 文件
    保存的内容包括：伴侣姓名、性格、会话名称、所有聊天记录

    调用时机：
    - 用户发送消息后
    - AI 回复完成后
    - 点击"新建会话"时
    """
    # 如果还没有当前会话名称，说明初始化还没完成，跳过
    if not st.session_state.get("current_session"):
        return

    # 1. 确保文件夹存在
    ensure_session_dir()

    # 2. 把要保存的数据组装成一个字典
    session_data = {
        "nick_name": st.session_state.nick_name,
        "character": st.session_state.character,
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages
    }

    # 3. 写入 JSON 文件
    #    ensure_ascii=False：中文不会被转成 \uXXXX，直接存中文
    #    indent=4：格式化输出，方便人直接打开看
    file_path = get_session_file_path(st.session_state.current_session)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)


def load_all_sessions():
    """
    扫描 session/ 文件夹，返回所有会话名称的列表
    按文件修改时间排序（最新的排在最前面）

    注意：这里只返回文件名（去掉 .json 后缀），不读取文件内容
    """
    ensure_session_dir()

    sessions_list = []
    # os.listdir：列出文件夹里所有文件
    for filename in os.listdir(SESSION_DIR):
        if filename.endswith(".json"):
            # filename 类似 "2026-05-29_14-30-00_a1b2c3.json"
            # filename[:-5] 去掉最后5个字符 ".json"，得到纯会话名
            sessions_list.append(filename[:-5])

    # 按文件修改时间倒序排列（最近用的排最上面）
    sessions_list.sort(
        key=lambda name: os.path.getmtime(get_session_file_path(name)),
        reverse=True
    )
    return sessions_list


def load_session(session_name):
    """
    从文件加载指定会话，恢复到当前界面

    做的事情：
    1. 读取对应的 JSON 文件
    2. 把里面的姓名、性格、聊天记录恢复到 st.session_state
    3. 界面会自动刷新，显示恢复后的聊天记录
    """
    file_path = get_session_file_path(session_name)

    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 把文件里的数据恢复到当前会话状态
            st.session_state.nick_name = data.get("nick_name", DEFAULT_NICK_NAME)
            st.session_state.character = data.get("character", DEFAULT_CHARACTER)
            st.session_state.current_session = session_name
            st.session_state.messages = data.get("messages", [])
        else:
            st.warning(f"会话文件不存在：{session_name}")
    except Exception:
        st.error(f"加载会话失败，文件可能已损坏：{session_name}")


def delete_session(session_name):
    """
    删除指定会话的 JSON 文件

    调用时机：用户在侧边栏点击 ❌ 按钮
    """
    file_path = get_session_file_path(session_name)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        st.error(f"删除会话失败：{session_name}")


def reset_to_new_session():
    """
    清空当前聊天，切换到全新的空会话
    会先生成一个新的会话名称，然后保存一个空的会话文件
    """
    st.session_state.messages = []
    st.session_state.current_session = generate_session_name()
    save_session()  # 立即保存一个空文件，让新会话出现在侧边栏列表中


# ============================================================
# 第七部分：初始化 —— 设置 Streamlit 会话状态
# ============================================================
# st.session_state 是 Streamlit 的"记忆"机制
# 页面刷新或重新运行时，里面的值会保留，不会丢失
# 每个用户的浏览器窗口有自己独立的 session_state

# 初始化聊天消息列表（空列表 = 还没有聊天记录）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化伴侣姓名
if "nick_name" not in st.session_state:
    st.session_state.nick_name = DEFAULT_NICK_NAME

# 初始化伴侣性格
if "character" not in st.session_state:
    st.session_state.character = DEFAULT_CHARACTER

# 初始化当前会话名称（首次打开自动生成一个新的）
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()

# ============================================================
# 第八部分：初始化 OpenAI 客户端（连接 DeepSeek API）
# ============================================================
# 从系统环境变量读取 API Key（不是直接写在代码里，更安全）
api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )
else:
    # 如果没有设置 API Key，client 设为 None
    # 后面聊天时会检测并给出提示，而不是直接报错崩溃
    client = None

# ============================================================
# 第九部分：页面主体 —— 标题和 Logo
# ============================================================

st.title("🥰 AI智能伴侣")           # 页面大标题
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH)              # 侧边栏顶部显示 Logo

# ============================================================
# 第十部分：侧边栏（左侧面板）
# ============================================================
# st.sidebar 里的所有内容都会显示在左侧
# 用一个 with 块把侧边栏的代码组织在一起，更清晰

with st.sidebar:
    # ---------- 区域一：控制面板 ----------
    st.subheader("📋 AI控制面板")

    # 【退出登录按钮】—— 点击后回到登录页
    if st.button("🚪 退出登录", use_container_width=True):
        save_session()
        st.session_state.logged_in = False
        st.rerun()

    # 【新建会话按钮】
    # st.button 返回 True 表示用户点击了它
    if st.button("💞 新建会话", use_container_width=True):
        save_session()          # 1. 先把当前聊天保存到文件
        reset_to_new_session()  # 2. 再切换到全新的空会话
        st.rerun()              # 3. 刷新页面，让界面显示新会话

    # 【会话历史列表】
    st.caption("📂 会话历史（点击可切换，❌ 可删除）")

    # 加载所有已保存的会话名称
    sessions_list = load_all_sessions()

    if not sessions_list:
        st.caption("暂无历史会话，开始聊天吧 ~")
    else:
        for session_name in sessions_list:
            # 把每一行分成两列：左边4份宽（放会话名按钮），右边1份宽（放删除按钮）
            col_left, col_right = st.columns([4, 1])

            with col_left:
                # 会话名按钮：点击后加载该会话
                # type="primary"=高亮显示当前选中的会话，其他用 secondary=灰色
                is_current = (session_name == st.session_state.current_session)
                btn_type = "primary" if is_current else "secondary"

                # 用 session_name 本身作为按钮文字
                if st.button(
                    session_name,
                    use_container_width=True,
                    icon="📄",
                    key=f"load_{session_name}",
                    type=btn_type
                ):
                    # 先保存当前会话，再加载选中的会话
                    save_session()
                    load_session(session_name)
                    st.rerun()

            with col_right:
                # 删除按钮：只有图标没有文字
                if st.button(
                    "",
                    use_container_width=True,
                    icon="❌",
                    key=f"delete_{session_name}",
                ):
                    delete_session(session_name)
                    # 如果删的是当前正在用的会话，自动切到新会话
                    if session_name == st.session_state.current_session:
                        reset_to_new_session()
                    st.rerun()

    # 分割线
    st.divider()

    # ---------- 区域二：伴侣信息设置 ----------
    st.subheader("👤 伴侣的信息")

    # 姓名输入框
    # value= 设置默认显示的值
    # 用户修改后，实时更新到 session_state
    new_name = st.text_input(
        "姓名",
        placeholder="请输入伴侣的姓名",
        value=st.session_state.nick_name
    )
    if new_name:
        st.session_state.nick_name = new_name

    # 性别选择（目前仅做展示，不影响 AI 行为）
    st.radio("性别", ["男", "女"], key="gender")

    # 性格输入框（多行文本）
    new_character = st.text_area(
        "性格",
        placeholder="请输入伴侣性格，例如：温柔可爱、喜欢撒娇...",
        value=st.session_state.character
    )
    if new_character:
        st.session_state.character = new_character


# ============================================================
# 第十一部分：聊天区域（页面中央）
# ============================================================

# 显示当前会话名称（方便你知道自己在哪个会话里）
st.caption(f"📌 当前会话：{st.session_state.current_session}")

# 遍历并显示所有历史消息
# msg["role"] 可以是 "user"（用户）或 "assistant"（AI 回复）
# st.chat_message 会根据 role 自动显示对应的头像和气泡样式
for msg in st.session_state.messages:
    if msg["role"] != "system":   # 跳过系统提示词，不显示给用户看
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# ============================================================
# 第十二部分：消息输入与 AI 回复
# ============================================================

# st.chat_input 在页面底部显示一个输入框
# 用户输入内容并按回车后，prompt 变量会拿到输入的文本
# := 是"海象运算符"，意思是：赋值的同时做判断
if prompt := st.chat_input("请输入您要问的问题"):

    # ---- 步骤1：显示用户消息，并加入聊天记录 ----
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_session()  # 立即保存，防止用户消息丢失

    # ---- 步骤2：调用 AI 大模型获取回复 ----
    # 检查 API Key 是否已设置
    if client is None:
        st.error("❌ 未设置 DEEPSEEK_API_KEY 环境变量，无法调用 AI。\n\n"
                 "请在终端执行：`export DEEPSEEK_API_KEY=你的密钥` 后重新运行。")
    else:
        try:
            # 在界面上显示 AI 头像，准备流式输出回复
            with st.chat_message("assistant"):
                # st.empty() 创建一个"占位符"，后续可以动态往里面填内容
                reply_placeholder = st.empty()

                # st.spinner 显示一个加载动画，告诉用户"正在思考"
                with st.spinner("💭 思考中..."):

                    # ---- 步骤2.1：构建发送给 AI 的消息列表 ----
                    # 动态生成系统提示词：把姓名和性格填入模板
                    system_prompt = SYSTEM_PROMPT_TEMPLATE % (
                        st.session_state.nick_name,
                        st.session_state.character
                    )
                    # 消息列表 = 系统提示词 + 所有历史聊天记录
                    api_messages = [
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ]

                    # ---- 步骤2.2：调用 DeepSeek API（流式模式） ----
                    # stream=True 表示"边生成边返回"，不用等 AI 全部想完
                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                        stream=True,
                        extra_body={"thinking": {"type": "enabled"}}  # 开启 DeepSeek 思考模式
                    )

                    # ---- 步骤2.3：逐块接收 AI 的回复 ----
                    thinking_content = ""   # 存放 AI 的"思考过程"
                    reply_content = ""      # 存放 AI 的"最终回复"

                    for chunk in stream:
                        # chunk 是 API 返回的一个小片段
                        delta = chunk.choices[0].delta

                        # reasoning_content：DeepSeek 的思考过程（像草稿纸）
                        if getattr(delta, "reasoning_content", None):
                            thinking_content += delta.reasoning_content

                        # content：AI 真正要输出的回复文字
                        if delta.content:
                            reply_content += delta.content
                            # 每收到一点新文字，就立即更新到页面上
                            reply_placeholder.markdown(reply_content)

                # ---- 步骤2.4：思考过程折叠展示 ----
                # 如果有思考内容，显示一个可展开的区域
                if thinking_content:
                    with st.expander("💭 查看思考过程"):
                        st.markdown(thinking_content)

                # ---- 步骤2.5：把 AI 回复加入聊天记录，并保存 ----
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply_content
                })
                save_session()

        except Exception as e:
            # 如果 API 调用出错（网络问题、密钥无效等），显示错误信息
            st.error(f"请求失败：{e}")
