# TripPilot · 基于 Qwen3 LoRA 的旅行工具调用智能体

TripPilot 是一个旅行 Function Calling 项目作品集，包含数据规范化、依赖式工具协议、LoRA 训练与评测代码，以及可在本地运行的旅行工作台。工作台展示需求补全、攻略与天气的顺序调用、酒店与评价串联、路线查询和历史模型预测。

**可直接运行的部分使用离线控制器和模拟工具，不需要模型权重、GPU 或 API Key。** LoRA 回放来自历史评测记录，不是实时模型推理。仓库附带嘉兴、杭州、上海三个城市的攻略样例。

## 安装与启动

### 环境要求

- Git，以及 Python 3.9 或更新版本；本地核验使用 Python 3.12.14。
- 支持现代 JavaScript 的浏览器。前端静态文件已在仓库内，不需要 Node.js 或构建步骤。
- 离线演示与单元测试只使用 Python 标准库，无需 `pip install`。克隆完成后，运行演示无需访问外部 API。
- 以下命令均从仓库根目录执行，除非明确写出 `cd`。

### macOS / Linux 命令行

```bash
git clone https://github.com/Gary-02-Gao/TripPilot.git
cd TripPilot
python3 --version
python3 -m venv .venv
.venv/bin/python 03_核心代码/demo_server.py --host 127.0.0.1 --port 8091
```

### Windows PowerShell

```powershell
git clone https://github.com/Gary-02-Gao/TripPilot.git
cd TripPilot
python --version
python -m venv .venv
.\.venv\Scripts\python.exe 03_核心代码/demo_server.py --host 127.0.0.1 --port 8091
```

出现 `Travel Agent Demo: http://127.0.0.1:8091` 后，打开 <http://127.0.0.1:8091>。保持终端运行，按 `Ctrl+C` 停止服务。命令行运行不依赖虚拟环境激活脚本。

macOS 也可在创建 Python 3.12 的 `.venv` 后双击根目录 `启动项目.command`；双击 `检查环境.command` 检查环境。启动器默认使用 18091，端口占用时在 18091–18110 内寻找空闲端口，并自动打开浏览器。`.command` 启动器要求 Python 3.10+，仅供 macOS 使用；其他系统使用上面的 Python 命令。迁移说明见 [本地运行说明](本地运行说明.md)。

## 使用与预期结果

在“旅行工作台”输入需求。每个独立例子开始前点击“新对话”，避免沿用上一个待补全请求。

| 操作或输入 | 可以核验的结果 |
| --- | --- |
| 输入“帮我规划一次旅行”，再输入“杭州” | 先追问城市、不调用工具；补充后依次出现 `search_travel_guide`、`get_weather_info` |
| 输入“嘉兴三日游” | 返回嘉兴三日安排，并展示攻略检索与天气查询参数 |
| 输入“推荐杭州500-800元的酒店” | 依次出现 `recommend_hotels`、`get_hotel_reviews`，显示模拟酒店与评价 |
| 输入“去上海博物馆怎么走” | 调用 `query_route`；使用预设上海起点与模拟路线数据 |
| 点击“打开归档 LoRA 样本” | 显示历史模型预测，并标明“非实时推理” |
| 切换“训练与评测” | 显示工具名准确率、严格匹配率及评测口径 |

工具卡片可以展开或收起参数。“新对话”清空服务端共享会话；重启服务也会清空会话。演示行程最多展示 5 天，未指定日期时使用服务启动日期之后第 7 天。天气、酒店、评价与路线为模拟数据。

## 配置

直接运行 `demo_server.py` 时，可使用命令行参数、环境变量，或可选文件 `03_核心代码/.env`。**优先级为命令行参数 > 已有环境变量 > `.env` > 内置默认值。** `.env` 位于核心代码目录，不是仓库根目录，仅使用简单的 `KEY=value` 格式。

| 参数 | 环境变量 | 内置默认值 | 说明 |
| --- | --- | --- | --- |
| `--host` | `DEMO_HOST` | `127.0.0.1` | 本地监听地址 |
| `--port` | `DEMO_PORT` | `8090` | 端口，须为 1–65535 的整数；上面的启动示例显式使用 8091 |

不创建 `.env` 也能按上面的命令运行。需要文件配置时，把 [配置样例](03_核心代码/.env.example) 复制为同目录的 `.env`：

```dotenv
DEMO_HOST=127.0.0.1
DEMO_PORT=8091
```

然后执行 `.venv/bin/python 03_核心代码/demo_server.py`（Windows 使用 `.\.venv\Scripts\python.exe`）。修改配置后重启服务。macOS 双击启动器读取 `.launcher/config.json` 并传入显式参数，因此使用该文件中的端口设置。`.env`、`.venv/` 和 `.runtime/` 已加入 Git 忽略规则。

## HTTP 接口

服务启动后，可在另一个终端检查接口。以下 `curl` 示例适用于 macOS / Linux；也可以直接在浏览器打开健康检查和评测地址。

```bash
curl --fail http://127.0.0.1:8091/health
curl --fail -X POST http://127.0.0.1:8091/api/reset
curl --fail http://127.0.0.1:8091/api/chat \
  -H 'Content-Type: application/json' \
  --data '{"message":"嘉兴三日游"}'
```

| 方法与路径 | 请求 / 响应 |
| --- | --- |
| `GET /health` | 当前数据返回 `{"status":"healthy","mode":"offline","guides":3}` |
| `POST /api/chat` | 请求为 `{"message":"旅行需求"}`；响应包含 `text`、`workflow`、`tools`、`needs_input` |
| `POST /api/reset` | 无需请求体；返回 `{"status":"ok"}` |
| `GET /api/evaluation` | 返回仓库附带的历史评测指标与归档样本 |

聊天请求体上限为 16,000 字节；非法 JSON、错误消息类型或超长请求返回 HTTP 400。服务使用单个共享会话，没有用户隔离、鉴权或持久化，适用于本地单用户演示。

## 测试与结果校验

无需下载训练依赖或模型。在仓库根目录执行：

```bash
cd 03_核心代码
../.venv/bin/python -B -m unittest discover -s tests -v
cd ..
.venv/bin/python -B 02_成果与验证/verify_results.py
```

Windows PowerShell 对应命令：

```powershell
cd 03_核心代码
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
cd ..
.\.venv\Scripts\python.exe -B 02_成果与验证/verify_results.py
```

预期为 **15 项测试通过**，以及 `PASS: 378 unique nodes / 200 conversations; all 10 counts match.`。单元测试覆盖需求补全、工具依赖顺序、路线参数契约、领域拒答、数据规范化和异常模型输出解析，并实际执行数据转换 CLI。计数复算核对已归档评分，不会重新调用模型。

交付文件 SHA-256 见 [MANIFEST.sha256](MANIFEST.sha256)；源码原始哈希、当前交付哈希及修改记录见 [source-manifest.json](02_成果与验证/source-manifest.json)。实机核验范围见 [本地环境验证](本地环境验证.md)。

## 数据规范化示例

仓库提供两条专门编写的合成对话和嘉兴、杭州、上海三城市编码映射，无需外部数据即可执行规范化流程。在仓库根目录运行（Windows 将 `.venv/bin/python` 换为 `.\.venv\Scripts\python.exe`）：

```bash
.venv/bin/python -c "from pathlib import Path; Path('.runtime').mkdir(exist_ok=True)"
.venv/bin/python 03_核心代码/canonicalize_sequential_dataset.py \
  --input 03_核心代码/examples/synthetic_conversations.json \
  --mapping 03_核心代码/examples/city_code_mapping.json \
  --output .runtime/canonicalized-example.json \
  --training
```

预期输出 `{"input_rows": 2, "output_rows": 3}`。第一条并行攻略/天气调用被拆为两个训练前缀，天气调用前已有攻略结果；城市编码规范为名称；路线参数移除旧版 `city_code`。生成文件位于已忽略的 `.runtime/`。样例只用于检验协议转换，不是原始训练集或模型效果证据；来源说明见 [样例说明](03_核心代码/examples/README.md)。

## 训练与评测

历史实验使用 Qwen3-0.6B、LoRA rank 32、BF16 与 DDP，仅最后一条 Assistant 消息参与 Loss；最终训练阶段记录 3,048 条顺序化样本。顺序化协议 v2 在 200 条对话、378 个 Assistant 节点上，以标准历史上下文逐节点独立预测：

| 指标 | 计数 | 结果 |
| --- | ---: | ---: |
| 工具名序列正确 | 163 / 169 | 96.45% |
| 工具调用严格匹配（名称与参数） | 155 / 169 | 91.72% |
| 预测工具输出 JSON 合法 | 183 / 183 | 100% |

评分定义、逐节点判定和三阶段适配器合并顺序见 [成果与验证](02_成果与验证/README.md)。这些是历史实验结果；本次仓库验证没有重新执行 GPU 训练或完整模型推理，自主多轮任务成功率尚未评测。

训练和模型重建属于独立运行路径：需另外准备兼容的 PyTorch、Transformers、PEFT 及其运行依赖，以及基座、完整训练/测试数据和三个适配器权重。本仓库未提供训练环境锁定文件或这些模型与完整数据资产，不能仅凭克隆仓库重现原始训练结果。

外部环境与资产齐备后，可在仓库根目录使用以下参数模板；`/path/to/...` 需替换成实际路径。评测数据需为 JSON 数组。

```bash
python3 03_核心代码/train_qwen_last_assistant_lora.py \
  --model_name_or_path /path/to/base-model \
  --train_file /path/to/train.json \
  --tools_file 03_核心代码/all_tools_sequential_v2.json \
  --output_dir /path/to/adapter-output \
  --local_files_only

python3 03_核心代码/evaluate_qwen_fc.py \
  --model-path /path/to/merged-model \
  --data-file /path/to/test.json \
  --tools-file 03_核心代码/all_tools_sequential_v2.json \
  --output-dir /path/to/evaluation-output \
  --local-files-only
```

训练与数据检查脚本使用下划线参数，评测脚本使用连字符参数。请显式指定本仓库的 `all_tools_sequential_v2.json`，不要依赖历史脚本中未附带的默认数据或工具文件。完整模型必须按 [model-chain.json](02_成果与验证/model-chain.json) 中的顺序逐阶段合并，每一步以上一步合并结果作为基座。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `01_项目演示/` | 一分钟字幕演示视频及工作台截图 |
| `02_成果与验证/` | 训练记录、公开评测、逐节点评分、复算脚本与来源哈希 |
| `03_核心代码/` | 数据处理、训练、模型合并、评测、离线控制器与 HTTP 服务 |
| `03_核心代码/tests/` | Python 标准库单元测试 |
| `03_核心代码/examples/` | 合成对话与三城市映射，用于本地数据规范化验收 |
| `03_核心代码/web/` | HTML / CSS / JavaScript 前端及评测数据 |
| `03_核心代码/rag-system/original_data/travel_guides/` | 三个城市的模型生成攻略样例 |
| `.launcher/` | macOS 本地启动配置与启动器 |

实现索引见 [核心代码说明](03_核心代码/README.md)，演示见 [一分钟项目视频](01_项目演示/TripPilot_60s_字幕版.mp4)。这是精选归档作品集，公开提交历史从仓库整理发布开始，不包含原始训练工程的完整开发历史。相关旧项目的核验与本次复用范围见 [源码来源与开发历史](02_成果与验证/development-provenance.md)。

## 常见问题

- **端口被占用：**改用 `--port 8092` 并打开对应地址；Python 命令行启动不会自动换端口，macOS 双击启动器会。
- **双击提示缺少环境：**先在仓库根目录创建 `.venv`；虚拟环境不随 Git 仓库发布，移动目录后应重建。
- **页面或攻略缺失：**保留完整目录结构，包括 `web/`、`web/evaluation.json` 和三个攻略文件，不要只复制 `demo_server.py`。
- **启动提示端口不是整数：**检查 `DEMO_PORT` 环境变量及 `03_核心代码/.env`，只填写数字；非法环境值也可能影响带命令行参数的启动。
- **训练脚本找不到模块、数据或权重：**它们不属于零依赖演示的运行条件，请按上节准备独立环境与外部资产。

## 许可

本仓库代码与随附文档采用 **MIT License**，完整条款位于根目录 [LICENSE](LICENSE)。外部依赖、基座模型及外部数据仍适用各自许可；本仓库不分发上述模型权重与完整训练数据。
