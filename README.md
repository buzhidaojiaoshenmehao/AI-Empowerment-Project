# AI 知识库

基于 **FastAPI + LangChain + 本地向量检索** 的项目知识沉淀系统，支持文档分类管理、可追溯智能对话、知识图谱、新人赋能、知识任务、飞书集成与离职交接。

> 普通用户、知识运营人员和项目管理员请先阅读：[AI 赋能项目使用手册](docs/USER_GUIDE.md)。

## 功能特性

- 📚 **知识库管理** — 文档、飞书和交接资料统一形成来源、知识资产与历史版本，支持分类、责任人、复审、生命周期和来源权限管理
- 💬 **知识对话** — 纯对话界面，RAG 增强检索 + 流式输出，可溯源参考来源
- 🔗 **知识图谱** — 按文档知识片段生成中文主题节点，结合文档结构与语义关系自动关联，支持手动重新构建
- 🧭 **新人赋能** — 按岗位生成学习路径、知识覆盖度、待补资料和优先阅读清单
- ✅ **知识任务** — 将新人缺口、资产复审、飞书异常和交接补充转为可分派、可提交证据、可独立验收的治理任务；交接补充与资产复审验收后原子回写精确关联业务对象
- 📢 **飞书集成** — 群消息自动归档、富文本解析、截图 OCR/多模态提取、知识候选聚合、机器人问答与知识推送
- 📋 **离职交接** — 8 项必交清单、证据验收、结构化风险、学习计划联动与不可变关闭快照
- 🖥️ **系统设置** — 运行监控、容量基线、LLM 配置、网络代理、数据治理与服务控制

## 快速开始

### 1. 克隆项目

```bash
git clone <你的 GitHub 仓库地址>
cd AI-Empowerment-Project
```

### 2. 安装依赖

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r backend/requirements.txt
```

> 项目使用 LangChain 1.x，需要 Python 3.10+；推荐 Python 3.12。首次安装后可执行 `python _check_deps.py` 核对运行环境。

### 3. 配置环境变量

在项目**根目录**创建 `.env` 文件（`config.py` 会自动读取）：

```ini
# ── 方式 A：使用 OpenAI ──
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

```ini
# ── 方式 B：使用 DeepSeek ──
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

```ini
# ── 方式 C：使用 MiniMax ──
OPENAI_API_KEY=your-minimax-key
OPENAI_BASE_URL=https://api.minimax.chat/v1
LLM_MODEL=minimax-abab6.5s-chat
```

```ini
# ── 方式 D：使用本地 Ollama（API Key 留空）──
OPENAI_API_KEY=
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2:7b
```

> 需要先安装 [Ollama](https://ollama.ai) 并拉取模型：`ollama pull qwen2:7b`

另外支持的可选配置：

```ini
# 向量模型（仅 OpenAI 系列有效，其他兼容服务自动使用本地哈希向量）
EMBEDDING_MODEL=text-embedding-ada-002

# 服务端口
APP_PORT=8000

# 每日治理调度扫描间隔（秒）；调度事实和任务仍持久化在 SQLite
GOVERNANCE_SCAN_SECONDS=300
```

> `config.json` 是系统设置页面保存配置时生成的本地运行时文件，可能包含 API Key 和飞书密钥，请不要提交到仓库。仓库只保留 `config.example.json` 作为格式示例。

### 4. 启动服务

**方式 A：一键启动（推荐）**

直接双击 `start.bat`，脚本会自动检查环境、安装依赖并启动服务。

**方式 B：命令行启动**

```bash
# 在项目根目录执行
.venv/bin/python run.py
```

服务默认启动在 `http://localhost:8000`，浏览器会自动打开。首次使用前由管理员创建成员并生成一次性激活链接，成员设置密码后登录。

本地比赛演示可在 `.env` 中设置 `DEMO_SIMPLE_AUTH=true`。启用后，项目成员会自动激活，登录密码为企业邮箱的 `@` 前缀，例如 `demo-member@example.com` 的密码为 `demo-member`。该模式仅用于受控演示，正式部署必须关闭并使用邀请激活与强密码策略。

### 5. 打开前端

浏览器直接访问 `http://localhost:8000`，前端页面由 FastAPI 自动托管。

## 页面功能说明

| 标签页 | 功能 |
|--------|------|
| 💬 **知识对话** | 纯对话界面，输入问题自动检索知识库，流式回答 |
| 📚 **知识库** | 单/批量上传文档（姓名 + 分类 + 文件）、分类 CRUD 管理、按分类展示文档、文件预览 |
| 🔗 **知识图谱** | 自动生成知识关系网络，节点/关系表格展示，支持手动重新构建 |
| 🧭 **新人赋能** | 按岗位输出学习路径、知识覆盖度、待补资料和优先阅读清单 |
| 📢 **飞书集成** | 群采集规则、文本/图片/富文本沉淀、异常重试、机器人问答、知识推送与审计 |
| 📋 **离职交接** | 必交项提交/退回/验收、结构化风险关闭、接替人确认、学习计划联动与快照审计 |
| ⚙️ **系统设置** | LLM 配置（多供应商切换）、网络代理、服务重启、运行状态 |

## API 配置（网页端）

启动后，在 ⚙️ **系统设置** 中填入：
1. **API Key** — 你的 API 密钥
2. **API 地址** — 可切换不同供应商（OpenAI / DeepSeek / MiniMax / Ollama 等）
3. **模型名称** — 如 `gpt-4o-mini`、`deepseek-chat` 等

配置会自动生效，无需重启服务。

### 本地 OCR 与飞书图片权限

推荐安装本地 OCR 可选依赖，以便在无外网或飞书 OCR 限流时继续识别中文聊天截图：

```bash
python -m pip install -r backend/requirements-ocr.txt
```

本地 OCR 使用 RapidOCR 与 ONNX Runtime，扫描 PDF 页面由 PyMuPDF 渲染，采用惰性初始化，只有收到待识别图片或扫描文档时才加载模型。该依赖完全可选；未安装时应用仍可正常启动，但纯扫描文档任务会明确失败并保留暂存文件，可在安装依赖并重启服务后从任务中心重试。

知识库异步上传支持 PNG、JPG、JPEG 和 WebP 图片。PDF 会优先使用原生文本，仅对无有效文字的页面执行 OCR；混合 PDF 即使个别扫描页无法识别，也会保留其他可用页面并在任务详情显示警告。单任务默认最多 OCR 100 页，可用 `MAX_DOCUMENT_OCR_PAGES` 调整。

飞书应用需要启用机器人能力，并申请以下权限：

- `im:resource`：下载用户消息及富文本中的图片原件。
- `optical_char_recognition:image`：调用飞书 OCR 提取截图文字。

系统按“本地 RapidOCR → 飞书 OCR → 当前 OpenAI 兼容多模态模型”的顺序识别图片。任一通道不可用、失败或未取得有效结果时会继续后续通道，并保留 `local_rapidocr`、`feishu_ocr` 或 `vision_model` 方法标识。全部失败时图片原件仍会保留，内容中心会标记为“需处理”并提供重试；错误信息会汇总，但不会包含已配置的密钥。

## API 文档

访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

### 核心 API

| 方法 | 路径 | 说明 |
|------|------|------|
| **分类管理** | | |
| GET | `/api/categories` | 获取所有分类 |
| POST | `/api/categories` | 添加分类 |
| PUT | `/api/categories` | 编辑分类 |
| DELETE | `/api/categories/{name}` | 删除分类 |
| **文档管理** | | |
| POST | `/api/documents/uploads` | 异步受理文档上传并返回持久任务 ID |
| POST | `/api/documents/upload-batches` | 批量受理新知识文件并返回各自独立的持久任务 |
| POST | `/api/documents/upload` | 兼容旧集成的同步上传接口 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/by-category` | 按分类返回文档 |
| GET | `/api/documents/preview/{filename}` | 预览文档内容 |
| DELETE | `/api/documents/{filename}` | 删除文档 |
| **检索 & 对话** | | |
| POST | `/api/search` | 检索知识库 |
| POST | `/api/chat` | 流式对话（RAG 增强） |
| POST | `/api/chat-with-sources` | 流式对话（带来源信息） |
| POST | `/api/smart-chat` | 智能模式（LangChain Agent 自动选择只读项目能力） |
| **离职交接** | | |
| POST | `/api/resignation/submit` | 离职交接（多文件上传） |
| GET | `/api/resignation/records` | 获取交接记录、进度和风险 |
| GET | `/api/resignation/{handover_id}` | 获取交接清单、风险、完成 Gate 与快照摘要 |
| POST | `/api/resignation/{handover_id}/accept` | 接替人确认接收 |
| POST | `/api/resignation/{handover_id}/items/{item_id}/actions` | 提交、退回、重提或验收必交项 |
| POST | `/api/resignation/{handover_id}/inventory/refresh` | 重新盘点离职人员关联知识与未完成任务 |
| POST | `/api/resignation/{handover_id}/inventory/{inventory_id}/actions` | 纳入、排除或重新审阅盘点项 |
| POST | `/api/resignation/{handover_id}/risks` | 新增结构化交接风险 |
| POST | `/api/resignation/{handover_id}/risks/{risk_id}/actions` | 缓解、申请关闭或确认关闭风险 |
| POST | `/api/resignation/{handover_id}/complete` | 管理角色校验全部 Gate 并关闭交接 |
| GET | `/api/resignation/{handover_id}/snapshot` | 读取不可变交接快照与校验值 |
| **新人赋能** | | |
| GET | `/api/onboarding/templates` | 返回当前项目岗位模板与必需知识主题 |
| PUT | `/api/onboarding/templates/{template_id}` | 治理角色维护岗位标准、权重和完成要求 |
| GET | `/api/onboarding/guide` | 按岗位和目标成员权限返回就绪度、缺口、推荐资料与当前计划 |
| GET | `/api/onboarding/knowledge-gaps` | 返回指定岗位的知识缺口 |
| GET | `/api/onboarding/plans` | 返回本人或授权目标成员的学习计划 |
| POST | `/api/onboarding/plans` | 为目标成员分配岗位学习计划 |
| PATCH | `/api/onboarding/plans/{plan_id}/items/{item_id}` | 更新阅读、实践证据或负责人确认 |
| GET | `/api/onboarding/handover` | 返回指定交接的新人接手摘要 |
| **知识任务** | | |
| GET | `/api/knowledge/tasks` | 按本人或治理范围返回任务、状态汇总和处理记录 |
| POST | `/api/knowledge/tasks` | 创建或按幂等键合并知识任务 |
| POST | `/api/knowledge/tasks/onboarding-gaps` | 将岗位知识缺口批量转换为任务 |
| GET | `/api/knowledge/tasks/{task_id}` | 获取授权范围内的任务详情 |
| POST | `/api/knowledge/tasks/{task_id}/actions` | 领取、分派、开始、提交、退回、验收、取消或备注 |
| POST | `/api/knowledge/tasks/{task_id}/notify` | 手动发送飞书提醒并记录送达结果 |
| **知识图谱** | | |
| GET | `/api/knowledge-graph` | 知识关系图谱 |
| POST | `/api/knowledge-graph/build` | 手动触发图谱构建 |
| **飞书集成** | | |
| POST | `/api/feishu/webhook` | 飞书回调入口 |
| POST | `/api/feishu/push/knowledge` | 推送每日知识卡片 |
| POST | `/api/feishu/messages/{message_id}/enrichments` | 异步重新识别图片/富文本并返回任务 ID |
| POST | `/api/feishu/messages/{message_id}/enrich` | 兼容旧集成的同步重新识别接口 |
| POST | `/api/feishu/candidates/batch-actions` | 异步批量分类、排除、重试或入库并返回任务 ID |
| POST | `/api/feishu/candidates/batch-action` | 兼容旧集成的同步批量处理接口 |
| **系统** | | |
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 知识库统计 |
| GET | `/api/ops/overview` | 管理员 SLO、组件健康、维护窗口、恢复演练和活动告警 |
| POST | `/api/ops/capacity/validate` | 提交隔离 SQLite 容量验证任务 |
| POST | `/api/ops/maintenance` | 登记计划维护窗口 |
| DELETE | `/api/ops/maintenance/{window_id}` | 取消计划维护窗口 |
| POST | `/api/ops/recovery-drills` | 提交隔离 SQLite 恢复演练任务 |
| GET | `/api/governance/targets/preview` | 预演来源、成员或单资产的定向治理范围 |
| GET | `/api/governance/targets/export` | 导出定向治理对象的结构化 JSON 副本并记录审计 |
| POST | `/api/governance/targets/delete` | 提交持久化定向删除任务，清除权威内容、原件和派生投影 |
| GET | `/api/governance/schedule` | 查看每日治理策略、下次窗口和近期持久调度意图 |
| POST | `/api/governance/schedule/scan` | 立即扫描并补齐当前应执行的每日治理任务 |
| POST | `/api/settings/llm` | 保存 LLM 配置 |
| POST | `/api/settings/network` | 保存网络代理配置 |
| POST | `/api/restart` | 重启服务 |

## 分类管理

系统内置两个默认分类：**交接文档**、**新人培训**。

- 在 📚 **知识库** 页面点击「分类管理」可进行增删改操作
- 上传文档时可手动选择分类，不选则由 AI 自动分类

## 离职交接流程

1. **填写交接信息** — 姓名、职能、接替人和计划完成日期（姓名与职能按当前登录身份自动填入，职能只读，取自本人账号资料）
2. **上传并归档资料** — 多文件强制归入「交接文档」，形成可追溯资产版本
3. **自动知识盘点** — 按稳定成员 ID 识别关联统一知识资产、未完成知识任务和图谱主题，接替人逐项纳入或说明排除
4. **补齐岗位清单** — 系统生成 8 项通用清单并追加岗位模板的知识主题、验收标准和实践要求
5. **验收与退回** — 接替人或管理角色逐项接收、退回并要求补充；退回自动创建或合并交接补充任务
6. **关闭结构化风险** — 离职人员或风险责任人登记等级、影响、责任人、截止时间、缓解措施并提交关闭证据，接替人或管理角色核对后确认关闭或重开；未关闭风险会阻断完成
7. **确认并生成计划** — 接替人确认后，系统按其自身权限创建或复用岗位学习计划并幂等加入交接资料
8. **完成 Gate** — 按必交项 60%、风险 20%、接替人确认 20% 展示进度，同时要求盘点全审阅、清单全验收、风险全关闭、接替人已确认且学习计划就绪
9. **封存快照** — 管理角色关闭后保存盘点结论、纳入资产版本、风险和确认记录的不可变快照，后续禁止修改交接事实

进度用于解释当前完成情况，不等于允许关闭；只有五个完成 Gate 同时成立才可封存。同一交接重复关闭返回既有快照，不重复生成业务结果。交接清单按“待办优先”展示：顶部聚合当前身份的待办并可定位，只默认展开含待办的分组和条目，已验收、已处理、已关闭记录折叠为摘要，排除原因与风险登记表单按需展开。

## 新人赋能

选择成员和岗位后，系统使用目标成员有权访问的当前有效知识资产，分别输出：

1. 项目知识就绪度：主题覆盖 50% + 权威性 20% + 新鲜度 20% + 可追溯性 10%
2. 个人上手进度：必读完成 60% + 实践任务 25% + 负责人确认 15%
3. 已覆盖和待补充的知识主题、证据与优先阅读资料
4. 可维护的岗位模板、持久学习计划、实践证据和负责人确认

缺少授权资料的阅读任务会被阻断，不能直接标记完成；实践任务必须提交证据。岗位模板更新不会静默改变进行中的计划。交接确认会自动生成或复用接替人计划；若历史接替人无法映射、岗位模板缺失或交接资料无权访问，交接记录会如实显示“计划待处理”并允许重试，不把部分完成显示为全部完成。

## 项目结构

```
ai-web/
├── backend/
│   ├── main.py                         # FastAPI 入口（路由）
│   ├── config.py                       # 配置（读取根目录 .env）
│   ├── requirements.txt                # Python 依赖
│   ├── feishu_bot.py                   # 飞书机器人集成
│   ├── storage/                         # SQLite、Repository、迁移与维护命令
│   ├── knowledge_base/
│   │   ├── document_loader.py          # 文档加载器
│   │   ├── vector_store.py             # 向量存储
│   │   ├── retriever.py                # 增强检索器
│   │   ├── llm_service.py              # 大模型服务
│   │   ├── knowledge_graph.py          # 知识关系图谱
│   │   ├── project_context.py          # 项目知识语境
│   │   └── __init__.py
│   └── uploads/                        # 上传文件目录（自动创建）
├── frontend/
│   ├── index.html                      # 单页应用（Vue 3）
│   ├── style.css                       # 样式
│   └── app.js                          # 前端逻辑
├── data/                               # SQLite、迁移报告与本地备份（不提交 Git）
├── categories.json                     # 旧版迁移输入（迁移后不再作为真相源）
├── docs/                               # 项目文档
├── chroma_db/                          # 本地向量数据持久化目录（自动创建）
├── run.py                              # 直接运行入口
├── build.bat                           # 打包 EXE 脚本
├── start.bat                           # 一键启动脚本
├── stop.bat                            # 停止服务脚本
├── AI-KB.spec                          # PyInstaller 打包配置
├── _check_deps.py                      # 依赖检查工具
├── runtime_hook_disable_telemetry.py   # PyInstaller 运行时钩子
└── README.md
```

## 成员与登录

当前单机试点支持两种认证模式。正式试点默认使用本地邀请制账号；比赛演示可设置 `DEMO_SIMPLE_AUTH=true`，成员创建后立即激活，初始密码为企业邮箱 `@` 前缀。管理员由部署人员创建，其他项目协作者默认使用 `project_member` 最小权限；后续可在系统设置的“项目成员与权限”中维护角色、状态和本人职能。职能作为账号档案字段持久化在 `users.duty`，随登录身份返回，并在离职登记时只读自动填充。

```bash
# 创建或更新成员关系
python3 -m backend.storage.cli provision-member \
  --email member@example.com --name 成员姓名 --role project_member

# 生成 72 小时有效的一次性激活文件；文件权限自动设为 600
python3 -m backend.storage.cli invite-member \
  --email member@example.com --output data/bootstrap/invitations/member.json

# 查看成员（不会输出密码或令牌）
python3 -m backend.storage.cli list-members
```

邀请模式下，激活文件位于被 Git 忽略的 `data/` 目录，只能通过安全渠道交给对应成员；密码至少 12 位并包含字母、数字和特殊字符。演示模式是明确的弱认证边界，只能用于本地演示，正式部署必须关闭。两种模式均保留连续 5 次失败锁定 15 分钟、12 小时会话、成员停用后立即撤销会话、业务 API 登录校验和写操作 CSRF；飞书事件回调和存活检查保持公开。

角色边界：

- `project_admin`：成员、系统配置、飞书、知识生命周期与交接的完整管理权限；私有或受限正文仍需来源 ACL 明确授权。
- `project_manager`：项目知识、飞书与交接管理，不可管理系统密钥或成员。
- `knowledge_operator`：知识管理与发布、图谱维护、业务审计，不可管理系统配置。
- `project_member`：读取授权知识、上传普通文档、使用新人赋能和提交/确认本人交接。

## SQLite 数据维护

业务状态统一保存在 `data/app.db`。上传原件仍位于 `uploads/`，向量和自动图谱是可由 SQLite 重建的派生数据；向量索引使用 SQLite `chunk_id` 保证版本间不互相覆盖，自动图谱在单实例启动时按当前有效知识重建。数据库和备份均被 `.gitignore` 排除。

文档、飞书和交接入口共用 `KnowledgeSource → KnowledgeAsset → AssetVersion` 生命周期。每个资产最多一个当前生效版本；向量写入是版本切换硬门槛，图谱、项目语境和新人就绪度失败时会标记为待修复，不会把部分同步伪装成全部成功。撤销会立即停止 RAG、图谱和新人赋能消费，同时按权限和保留策略保留来源、历史版本与审计证据。

知识对话内置项目级版本化评测集，记录期望要点、允许资产/来源、无答案边界和检索策略快照。默认项目已维护 30 道当前知识真实问题，覆盖事实、流程、决策、风险、交接和无答案；基线运行 30/30 通过，54 条引用全部有效、权限泄露为 0、无答案准确率为 100%。PDF/OCR 兼容汉字会先按 NFKC 规范化再进入嵌入、词法检索和证据覆盖判断，索引指纹变化时从 SQLite 权威知识块重建。

```bash
# 查看 schema 与旧数据迁移状态
python3 -m backend.storage.cli status

# 检查数据库完整性和外键
python3 -m backend.storage.cli quick-check

# 创建在线一致性备份
python3 -m backend.storage.cli backup

# 从备份恢复；必须先停止 Web 服务，命令会检测运行锁并拒绝在线恢复
python3 -m backend.storage.cli restore data/backups/app-YYYYMMDD-HHMMSS.db
```

数据库路径通过环境变量或部署配置管理，不由系统设置页面修改。当前单机试点只运行一个应用实例；应用会持有跨进程数据库锁，误启第二个实例会拒绝启动。登录、项目角色和来源权限已经落地；企业 SSO、组织目录和多项目切换仍属于后续阶段。

处理任务中心以文档名、消息来源或业务操作名作为主标题，技术任务 ID 仅用于详情诊断；服务端按当前身份和任务状态返回允许的取消或重试动作，前端不自行推断权限。任务搜索同时支持文档名称。

图谱重建、统一知识投影修复、飞书群历史同步、知识任务提醒、单/批量文档上传、扫描 PDF/图片 OCR、飞书候选批量处理和飞书图片/富文本 OCR 已经使用 SQLite 持久处理任务：请求快速返回任务 ID，后台记录阶段、进度、尝试、错误与结果，支持自动/人工重试、取消和重启恢复。上传原文先进入权限收紧的 `uploads/.staging`，执行前按当前成员与资产权限复核；成功后删除，失败时保留供人工重试，排队取消后清理并要求重新上传。任务发起人可人工重试本人失败任务，治理角色可重试项目范围任务，每次执行仍重新校验当前权限。永久删除保留审计墓碑，但同名文件重新上传会创建新的来源与资产身份，不会与墓碑冲突或复活旧记录。新建知识单批默认最多 20 个文件、总计 100MB，每个文件拥有独立任务，单项失败、重试或取消不会阻断其他文件；更新现有知识继续使用单文件入口。飞书候选批量处理最多 100 项，单项失败不阻断后续候选，取消前已完成项仍保留逐项结果。飞书 Webhook 先保存待识别记录并提交任务，后台再执行资源下载、本地 RapidOCR、飞书 OCR、多模态回退、候选重评和自动入库；人工重试执行前重新校验发布权限。治理角色可查看全部项目任务，普通成员仅可查看本人发起的任务。交接补充任务验收后在同一 SQLite 事务中精确重提交关联交接项，不替代接替人验收；资产复审任务验收后仅确认仍待复审且当前版本有效的精确资产，失效或撤销状态不会被旧任务恢复，并在同一事务创建按生命周期版本幂等的投影修复任务。草稿资产提交审核会为最新 `ready` 版本创建精确发布任务，独立且具备发布权限的验收人通过后，任务完成、绑定版本发布和投影修复入队在同一事务提交；退回草稿、生命周期变化或出现更新候选时旧任务只记录跳过，不覆盖新事实。项目级任务提醒默认关闭；启用后按静默窗口、频率限制和逾期等级扫描，把同轮任务合并成飞书摘要，并由持久处理任务完成自动重试和重启恢复。发送前会再次核对任务是否仍未结束且截止时间仍匹配，任务详情可查看排队、重试、失败、发送或跳过证据。旧 `/api/documents/upload`、`/api/feishu/candidates/batch-action` 与 `/api/feishu/messages/{message_id}/enrich` 同步合同继续保留给已有集成，产品界面使用对应异步入口。

项目管理员可在“系统设置 / 项目数据治理”配置飞书原文、访问审计、处理任务和普通备份保留策略。清理前先预演影响范围，输入确认短语后由处理任务执行；旧飞书原文仅在未关联候选或知识资产时脱敏，关键资产历史和变更审计不会自动删除。工作台还支持按知识来源、项目成员或单个知识资产预演影响、先导出 JSON 留档，再以精确确认短语提交永久删除。删除任务先让权威资产不可检索，再清除正文、受控目录原件、向量、图谱和上下文；成员删除同时停用项目身份并撤销会话及外部身份，治理运行与必要审计保留。每日调度默认按 `Asia/Shanghai` 02:00 创建并校验一致性备份，自动保留期清理默认关闭；管理员可调整 IANA 时区与时间、立即扫描，并查看近期调度对应的处理任务。调度意图按本地日期永久去重，服务错过窗口后会在下次启动补跑。数据库备份在线创建并校验，恢复仍需先停止 Web 服务后运行 `python -m backend.storage.cli restore <backup.db>`。

“系统设置 / 运行监控与容量基线”按 1 小时、24 小时、7 天或 30 天汇总普通接口 P95、异步受理、飞书事件持久化、RAG 首次内容、文档 5 分钟完成/明确失败率和任务故障 1 分钟可见率；月度可用性固定按 Asia/Shanghai 周一至周五 09:00–18:00 计算，并单列排除已登记的计划维护分钟。指标采用内存分钟聚合或 SQLite 事务事件，不记录请求正文、问题、Prompt、Cookie 或密钥；样本不足明确显示“采集中”。管理员可以发起隔离的 10,000 条消息 SQLite 查询容量验证，也可把最新有效备份恢复到临时库并校验哈希、完整性、外键和 schema；恢复演练不会替换在线数据库，存在有效备份且 30 天内无演练时服务启动会自动安排一次。演练报告与操作者、备份、RPO/RTO 和最小化校验摘要在同一事务写入安全审计。

## 架构说明

```
上传文档 → 文档加载器 → 文本分块 → 向量嵌入 → 本地向量库
                                                        ↓
用户提问 → 增强检索器(相似度+MMR) → 上下文拼接 → LLM → 流式回答
```

## 常见问题

**Q: 启动后页面显示 "LLM 未配置"？**
A: 点击 ⚙️ 系统设置，填入并保存 API Key。配置会写入本地 `config.json`，重启服务后仍会恢复。

**Q: 如何切换不同的模型供应商？**
A: 在系统设置中修改 API 地址和模型名称即可，支持 OpenAI / DeepSeek / MiniMax / Ollama 等兼容接口。

**Q: 上传文档时分类怎么选？**
A: 可以在知识库页面的「分类管理」中自定义分类列表，上传时从下拉列表选择。不选则由 AI 自动分类。

**Q: 如何给知识任务补充资料？**
A: 在知识任务详情的「完成证据」下点击「上传任务资料」。弹窗与知识库的“新建知识”使用同一套分类、格式和批量上传规则，但不会提供“更新现有知识”。文件受理后会进入处理任务中心，并以“知识库处理中”写入任务证据；确认处理任务成功后，再提交任务验收。

**Q: 离职交接上传的文件存在哪里？**
A: 交接文档自动归入「交接文档」分类，同时以离职人员姓名为作者存入知识库，可在知识库按分类查阅。
