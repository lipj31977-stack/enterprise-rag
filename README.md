# Enterprise Knowledge Base Copilot

> 基于检索增强生成（RAG）技术的企业知识库智能问答系统

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.1-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术栈](#3-技术栈)
4. [项目结构](#4-项目结构)
5. [环境要求](#5-环境要求)
6. [快速开始](#6-快速开始)
7. [配置说明](#7-配置说明)
8. [API 参考](#8-api-参考)
9. [常见问题排查](#9-常见问题排查)
10. [扩展方向](#10-扩展方向)
11. [许可证](#11-许可证)

---

## 1. 项目概述

**Enterprise Knowledge Base Copilot** 是一个面向企业内部知识管理场景的智能问答系统，基于 **检索增强生成（Retrieval-Augmented Generation，RAG）** 技术构建。系统允许用户将私有文档（PDF、TXT、Markdown 等格式）上传至知识库，并通过自然语言进行语义检索与问答，每条回答均附带可追溯的原文引用来源。

**核心能力：**

- **文档管理**：支持 PDF、TXT、Markdown 格式文档的上传、解析与管理
- **语义检索**：基于向量相似度的语义搜索，突破关键词匹配的局限
- **智能问答**：结合检索结果与大语言模型，生成有据可查的精确回答
- **引用溯源**：每条回答均标注所引用的文档原文片段及来源文件

本项目采用前后端分离架构，后端通过 FastAPI 暴露 RESTful 接口，前端以 Streamlit 快速构建交互界面，适合作为企业内部知识管理工具或 RAG 技术的学习参考实现。

---

## 2. 系统架构

### 2.1 RAG 工作流程

```
用户输入的自然语言问题
        │
        ▼
┌───────────────────────┐
│  Qwen Embedding 模型  │  ← 将问题转化为高维稠密向量（语义表示）
│  text-embedding-v3    │
└───────────────────────┘
        │  查询向量
        ▼
┌───────────────────────┐
│     FAISS 向量索引    │  ← 在本地向量库中执行近似最近邻（ANN）检索
│   (本地持久化存储)    │
└───────────────────────┘
        │  Top-K 相关文档片段
        ▼
┌───────────────────────┐
│  DeepSeek Chat 模型   │  ← 以检索结果作为上下文，生成最终回答
│   deepseek-chat       │
└───────────────────────┘
        │
        ▼
回答文本 + 原文引用来源
```

### 2.2 文档入库流程

```
用户上传文档
     │
     ▼
文件解析（PDF / TXT / MD）
     │
     ▼
文本切分（Chunk Size: 500，Overlap: 50）
     │
     ▼
Embedding 向量化（Qwen text-embedding-v3）
     │
     ├──► 向量存入 FAISS 索引（data/vector_store/）
     │
     └──► 文档元数据写入 SQLite 数据库（data/kb_copilot.db）
```

### 2.3 服务交互关系

```
┌──────────────────┐         HTTP REST          ┌──────────────────────┐
│  Streamlit 前端  │  ─────────────────────────► │   FastAPI 后端       │
│  localhost:8501  │ ◄─────────────────────────  │   localhost:8000     │
└──────────────────┘                             └──────────────────────┘
                                                          │
                              ┌───────────────────────────┼────────────────────────┐
                              │                           │                        │
                              ▼                           ▼                        ▼
                     ┌──────────────┐         ┌──────────────────┐     ┌──────────────────┐
                     │    SQLite    │         │  FAISS 向量索引  │     │  外部 AI 模型 API │
                     │  kb_copilot  │         │  (本地文件)      │     │  DeepSeek / Qwen  │
                     │    .db       │         │                  │     │                  │
                     └──────────────┘         └──────────────────┘     └──────────────────┘
```

---

## 3. 技术栈

| 层级             | 组件                     | 版本    | 说明                                       |
| ---------------- | ------------------------ | ------- | ------------------------------------------ |
| **前端界面**     | Streamlit                | 1.38.0  | 纯 Python 的 Web UI 框架，无需前端开发经验 |
| **后端框架**     | FastAPI + Uvicorn        | 0.115.0 | 高性能异步 RESTful API 框架                |
| **AI 编排框架**  | LangChain                | 0.3.1   | 负责 Prompt 构建、检索链与模型调用的编排   |
| **对话大模型**   | DeepSeek Chat            | —       | 负责理解上下文并生成自然语言回答           |
| **向量嵌入模型** | Qwen text-embedding-v3   | —       | 将文本转化为向量，支持中英文双语语义检索   |
| **向量数据库**   | FAISS（CPU 版）          | 1.8.0   | Meta 开源的高效本地向量索引库              |
| **关系型数据库** | SQLite（via SQLAlchemy） | 2.0.35  | 零配置嵌入式数据库，数据以单文件形式持久化 |
| **文档解析**     | pypdf                    | 4.3.1   | PDF 文件解析                               |
| **配置管理**     | pydantic-settings        | 2.5.2   | 基于 Pydantic 的类型安全配置加载           |
| **日志系统**     | loguru                   | 0.7.2   | 结构化日志记录                             |
| **Python 环境**  | Conda                    | —       | 虚拟环境隔离，推荐使用 Python 3.11         |

> **为什么同时使用 DeepSeek 和 Qwen？**  
> DeepSeek 目前未提供文本嵌入（Embedding）模型，而 RAG 的语义检索阶段必须依赖 Embedding 模型。本项目采用阿里云 Qwen 的 `text-embedding-v3` 模型弥补这一缺口，两者均提供免费调用额度。

---

## 4. 项目结构

```text
enterprise-kb-copilot/
│
├── backend/                        # 后端服务（FastAPI）
│   ├── main.py                     # 应用入口，生命周期管理，路由注册
│   ├── config.py                   # 基于 pydantic-settings 的配置管理
│   ├── database.py                 # SQLite 数据库连接与初始化
│   ├── models.py                   # SQLAlchemy ORM 数据表模型定义
│   ├── schemas.py                  # Pydantic 请求/响应数据结构定义
│   ├── routers/
│   │   ├── documents.py            # 文档管理接口（上传、列表、删除）
│   │   └── chat.py                 # 问答接口（RAG 全流程）
│   ├── services/
│   │   ├── document_service.py     # 文档解析 → 切分 → 向量化入库的业务逻辑
│   │   ├── vector_service.py       # FAISS 索引管理（调用 Qwen Embedding）
│   │   └── chat_service.py         # RAG 问答链（调用 DeepSeek Chat）
│   └── utils/
│       ├── file_parser.py          # PDF / TXT / Markdown 文件解析器
│       └── logger.py               # 日志配置
│
├── frontend/                       # 前端界面（Streamlit）
│   ├── app.py                      # 应用首页与导航配置
│   └── pages/
│       ├── 1_💬_Chat.py            # 问答对话页面
│       └── 2_📄_Documents.py       # 文档管理页面
│
├── data/                           # 运行时数据目录（首次启动后自动生成）
│   ├── uploads/                    # 用户上传的原始文档
│   ├── vector_store/               # FAISS 向量索引持久化文件
│   └── kb_copilot.db               # SQLite 数据库文件
│
├── logs/                           # 应用运行日志（自动生成）
├── scripts/                        # 辅助脚本
├── .env                            # 本地环境变量配置（含 API Key，禁止提交至版本库）
├── .env.example                    # 环境变量配置模板
├── .gitignore                      # Git 忽略规则（已包含 .env）
├── requirements.txt                # Python 依赖清单
├── Dockerfile                      # Docker 镜像构建文件
├── docker-compose.yml              # Docker Compose 编排配置
└── README.md                       # 项目文档（本文件）
```

---

## 5. 环境要求

| 依赖项           | 版本要求 | 说明                                |
| ---------------- | -------- | ----------------------------------- |
| Python           | 3.11     | 推荐通过 Conda 管理                 |
| Conda            | 任意版本 | 用于创建隔离的 Python 虚拟环境      |
| DeepSeek API Key | —        | 用于对话生成，获取地址见下文        |
| Qwen API Key     | —        | 用于文本向量化（阿里云 DashScope）  |
| 网络连接         | —        | 需能访问 DeepSeek 和阿里云 API 端点 |

### API Key 申请

| 服务        | 用途                              | 申请地址                                                             |
| ----------- | --------------------------------- | -------------------------------------------------------------------- |
| DeepSeek    | 对话生成（`deepseek-chat`）       | [platform.deepseek.com](https://platform.deepseek.com)               |
| 阿里云 Qwen | 文本向量化（`text-embedding-v3`） | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |

> 两个平台均提供一定量的免费调用额度，注册后即可直接使用。

---

## 6. 快速开始

### 步骤 1：克隆仓库

```bash
git clone <仓库地址>
cd enterprise-kb-copilot
```

---

### 步骤 2：创建并激活 Conda 虚拟环境

```bash
# 创建独立的 Python 3.11 环境（仅需执行一次）
conda create -n kb_copilot python=3.11 -y

# 激活环境（每次打开新终端均需执行）
conda activate kb_copilot
```

激活成功后，终端提示符的最左侧将显示 `(kb_copilot)` 标识。

---

### 步骤 3：安装 Python 依赖

```bash
pip install -r requirements.txt
```

若网络访问 PyPI 速度较慢，可使用国内镜像源加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 步骤 4：配置环境变量

**（1）复制配置模板：**

```bash
# Windows（PowerShell）
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

**（2）编辑 `.env` 文件，填入实际的 API Key：**

```ini
# ── 对话模型（DeepSeek） ────────────────────────────────────
OPENAI_API_KEY=sk-你的DeepSeek-API-Key
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat

# ── 向量嵌入模型（阿里云 Qwen） ────────────────────────────
EMBEDDING_API_KEY=sk-你的阿里云DashScope-API-Key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v3
```

> **⚠️ 安全提示**：`.env` 文件包含私密凭据，切勿将其提交至 Git 仓库。`.gitignore` 已默认忽略该文件。

---

### 步骤 5：启动后端服务

在终端 A 中执行（确保 `kb_copilot` 环境已激活）：

```bash
python -m backend.main
```

观察到如下输出，表明后端启动成功：

```text
============================================================
🚀 企业知识库 Copilot 后端启动中...
📁 数据目录已就绪: uploads=./data/uploads, vector_store=./data/vector_store
✅ 数据库初始化完成
✅ 后端启动完成，监听 0.0.0.0:8000
📖 API 文档: http://localhost:8000/docs
============================================================
```

首次启动时，系统将自动创建 `data/` 目录及 SQLite 数据库文件 `data/kb_copilot.db`，无需任何手动初始化操作。

---

### 步骤 6：启动前端界面

在终端 B 中执行（同样激活 `kb_copilot` 环境）：

```bash
cd frontend
streamlit run app.py
```

观察到如下输出，表明前端启动成功：

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

### 步骤 7：使用系统

1. 打开浏览器，访问 **http://localhost:8501**
2. 进入 **📄 Documents** 页面，上传 PDF 或 TXT 文档
3. 等待文档状态变为 `completed`（后台完成解析、切分与向量化）
4. 进入 **💬 Chat** 页面，以自然语言提问
5. 系统将返回 AI 生成的回答，并附带所引用的原文片段及来源文档名称

---

## 7. 配置说明

所有配置项均通过项目根目录的 `.env` 文件进行管理，由 `backend/config.py` 加载并提供给各模块使用。

| 配置项                 | 默认值                                              | 说明                                               |
| ---------------------- | --------------------------------------------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`       | —                                                   | DeepSeek Chat 的 API Key（必填）                   |
| `OPENAI_BASE_URL`      | `https://api.deepseek.com`                          | DeepSeek API 端点地址                              |
| `LLM_MODEL_NAME`       | `deepseek-chat`                                     | 对话模型名称                                       |
| `EMBEDDING_API_KEY`    | —                                                   | 阿里云 DashScope 的 API Key（必填）                |
| `EMBEDDING_BASE_URL`   | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Qwen Embedding API 端点地址                        |
| `EMBEDDING_MODEL_NAME` | `text-embedding-v3`                                 | 向量嵌入模型名称                                   |
| `BACKEND_HOST`         | `0.0.0.0`                                           | 后端监听地址                                       |
| `BACKEND_PORT`         | `8000`                                              | 后端监听端口                                       |
| `LOG_LEVEL`            | `INFO`                                              | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`） |
| `CHUNK_SIZE`           | `500`                                               | 文档切分的单块字符数上限                           |
| `CHUNK_OVERLAP`        | `50`                                                | 相邻文本块之间的重叠字符数                         |
| `RETRIEVER_TOP_K`      | `4`                                                 | 语义检索返回的最相关文档块数量                     |
| `UPLOAD_DIR`           | `./data/uploads`                                    | 原始文档存储路径                                   |
| `VECTOR_STORE_DIR`     | `./data/vector_store`                               | FAISS 索引持久化存储路径                           |

> **调优提示**：若回答质量不满足预期，可尝试适当增大 `RETRIEVER_TOP_K`（如调至 6 或 8），以向语言模型提供更多参考上下文；但过大的值会增加 API Token 消耗和延迟。

---

## 8. API 参考

后端启动后，可通过 **http://localhost:8000/docs** 访问 Swagger UI，支持在线浏览和调试全部接口。

### 系统接口

| 接口路径         | 方法 | 说明                                         |
| ---------------- | ---- | -------------------------------------------- |
| `/`              | GET  | 返回服务基本信息（名称、版本号、文档地址）   |
| `/api/v1/health` | GET  | 健康检查，返回数据库连接状态与向量库加载状态 |

### 文档管理接口

| 接口路径                   | 方法   | 说明                                           |
| -------------------------- | ------ | ---------------------------------------------- |
| `/api/v1/documents/upload` | POST   | 上传文档，后台异步执行解析 → 切分 → 向量化入库 |
| `/api/v1/documents`        | GET    | 查询已上传的文档列表及各文档处理状态           |
| `/api/v1/documents/{id}`   | DELETE | 删除指定文档（同步删除向量索引与数据库记录）   |

### 问答接口

| 接口路径                | 方法 | 说明                                            |
| ----------------------- | ---- | ----------------------------------------------- |
| `/api/v1/chat`          | POST | 执行 RAG 完整流程，返回回答文本及引用的原文片段 |
| `/api/v1/chat/sessions` | GET  | 查询所有历史对话会话                            |

---

## 9. 常见问题排查

### 问题 1：`conda: command not found` 或 `conda` 命令无响应

**原因**：Conda 未安装，或安装后未加入系统 `PATH` 环境变量。

**解决方案**：下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，安装过程中勾选 **"Add Miniconda3 to my PATH environment variable"** 选项，安装完成后重新打开终端。

---

### 问题 2：`ModuleNotFoundError: No module named 'xxx'`

**原因**：当前终端未激活 `kb_copilot` 虚拟环境，或依赖包未完整安装。

**解决方案**：

```bash
conda activate kb_copilot
pip install -r requirements.txt
```

---

### 问题 3：上传文档后报错 `Error code: 400` 或 `InvalidParameter`

**原因**：阿里云 Qwen Embedding API 的调用参数格式不兼容。

**解决方案**：

1. 确认 `.env` 中 `EMBEDDING_API_KEY` 已正确填写
2. 确认 `EMBEDDING_MODEL_NAME=text-embedding-v3`（注意拼写）
3. 重启后端服务：
   ```bash
   # 按 Ctrl+C 停止现有进程，然后重新启动
   python -m backend.main
   ```

---

### 问题 4：AI 回答"无法找到相关信息"或答非所问

**原因**：可能是文档未完成处理，或检索相关度较低。

**排查步骤**：

1. 在 **Documents** 页面确认文档状态为 `completed`
2. 确认提问内容与文档主题相关
3. 将 `.env` 中的 `RETRIEVER_TOP_K` 适当调大（例如从 `4` 调至 `8`）后重启后端

---

### 问题 5：`streamlit run app.py` 提示 `File does not exist`

**原因**：未在 `frontend/` 目录下执行命令。

**解决方案**：

```bash
cd frontend
streamlit run app.py
```

---

### 问题 6：后端启动后，前端请求返回连接错误

**原因**：前端指向的后端地址与实际监听地址不一致。

**解决方案**：确认 `.env` 中 `BACKEND_URL=http://localhost:8000`（本地开发时不应使用 `http://backend:8000`，该地址仅适用于 Docker Compose 部署环境）。

---

## 10. 扩展方向

以下为本项目可进一步扩展的功能方向，供有意深入的开发者参考：

- **流式输出（Streaming）**：通过 SSE（Server-Sent Events）实现逐字回显，提升交互体验
- **多知识库支持**：实现知识库的命名空间隔离，支持按知识库范围进行检索
- **用户认证**：引入 JWT（JSON Web Token）机制，实现用户注册、登录与权限管理
- **更多文档格式**：扩展解析器以支持 Word（`.docx`）、Excel（`.xlsx`）、HTML 等格式
- **Agent 化改造**：集成 LangGraph，构建具备工具调用能力的 RAG Agent
- **生产级部署**：配合已有的 `Dockerfile` 与 `docker-compose.yml` 进行容器化部署

---

## 11. 许可证

本项目基于 [MIT License](LICENSE) 开源，允许自由使用、修改与分发。
