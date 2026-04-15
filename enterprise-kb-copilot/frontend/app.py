"""
Streamlit 前端 — 首页

展示项目介绍、系统状态、知识库统计。
作为多页面应用的入口和导航中心。
"""

import streamlit as st
from utils import health_check, get_document_stats


def main():
    # ========== 页面配置 ==========
    st.set_page_config(
        page_title="企业知识库 Copilot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ========== 自定义样式 ==========
    st.markdown("""
    <style>
        /* 渐变标题 */
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #6c757d;
            margin-top: 0.3rem;
            margin-bottom: 2rem;
        }
        /* 功能卡片 */
        .feature-card {
            background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border-left: 4px solid #667eea;
            margin-bottom: 0.8rem;
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
        .feature-card h3 {
            margin: 0 0 0.5rem 0;
            color: #333;
            font-size: 1.1rem;
        }
        .feature-card p {
            margin: 0;
            color: #555;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        /* 状态指示器 */
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-green { background-color: #22c55e; }
        .status-red { background-color: #ef4444; }
        .status-yellow { background-color: #f59e0b; }
        /* 步骤编号 */
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.markdown("### 🤖 企业知识库 Copilot")
        st.caption("v1.0.0")
        st.markdown("---")

        # 系统状态
        status = health_check()
        is_healthy = status.get("status") == "healthy"
        db_ok = status.get("database") == "connected"
        vs_status = status.get("vector_store", "unknown")

        st.markdown("**🔌 系统状态**")

        if is_healthy:
            st.markdown(
                '<span class="status-dot status-green"></span> 服务运行中',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-red"></span> 服务未连接',
                unsafe_allow_html=True,
            )

        if db_ok:
            st.markdown(
                '<span class="status-dot status-green"></span> 数据库已连接',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-red"></span> 数据库未连接',
                unsafe_allow_html=True,
            )

        vs_color = "green" if vs_status == "loaded" else "yellow"
        vs_label = "向量库已加载" if vs_status == "loaded" else "向量库为空"
        st.markdown(
            f'<span class="status-dot status-{vs_color}"></span> {vs_label}',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**📖 功能导航**")
        st.page_link("pages/1_💬_Chat.py", label="💬 智能问答", icon="💬")
        st.page_link("pages/2_📄_Documents.py", label="📄 文档管理", icon="📄")

    # ========== 主区域 — 标题 ==========
    st.markdown(
        '<p class="main-title">🤖 企业知识库 Copilot</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">基于 RAG 的智能文档问答系统 — 上传文档，即刻获取精准回答</p>',
        unsafe_allow_html=True,
    )

    # ========== 知识库统计（仅在后端连接时显示） ==========
    if is_healthy:
        stats_resp = get_document_stats()
        if stats_resp.get("code") == 200:
            stats = stats_resp.get("data", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📄 文档总数", stats.get("total_documents", 0))
            with c2:
                st.metric("🧩 文本块数", stats.get("total_chunks", 0))
            with c3:
                st.metric("📐 向量总数", stats.get("total_vectors", 0))
            with c4:
                st.metric("✅ 已完成", stats.get("completed", 0))

    st.markdown("---")

    # ========== 功能介绍 ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📤 文档上传</h3>
            <p>支持 PDF、TXT、Markdown 格式。<br/>上传后自动解析、切分、向量化入库。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 智能检索</h3>
            <p>基于 FAISS 向量相似度检索，<br/>快速找到与问题最相关的文档片段。</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💡 精准回答</h3>
            <p>大模型生成回答并附带引用来源，<br/>所有回答均可溯源验证。</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== 快速开始 ==========
    st.markdown("### 🚀 快速开始")

    st.markdown("""
    <p>
        <span class="step-num">1</span> 进入 <strong>📄 文档管理</strong> 页面，上传你的文档<br/>
        <span class="step-num">2</span> 系统自动完成 解析 → 切分 → 向量化<br/>
        <span class="step-num">3</span> 进入 <strong>💬 智能问答</strong> 页面，输入问题获取精准回答
    </p>
    """, unsafe_allow_html=True)

    # ========== 技术栈 ==========
    st.markdown("---")
    st.markdown("### 🛠️ 技术栈")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown("**前端**\n\nStreamlit")
    with t2:
        st.markdown("**后端**\n\nFastAPI")
    with t3:
        st.markdown("**AI 框架**\n\nLangChain")
    with t4:
        st.markdown("**向量库**\n\nFAISS")
    with t5:
        st.markdown("**数据库**\n\nMySQL 8.0")


if __name__ == "__main__":
    main()
