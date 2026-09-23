# 数据规范化合成样例

本目录提供两条新编写的合成训练前缀，用于运行和检查数据规范化 CLI。它们不包含原始训练/测试数据、真实用户画像或真实工具返回，也不用于证明历史模型指标。路线坐标沿用离线 Demo 的公开预设值，不代表用户位置。

`synthetic_conversations.json` 故意保留规范化前的协议：城市使用编码、攻略与天气在同一次 Assistant 消息中调用、路线包含旧版可选参数 `city_code`。这些都是转换器需要处理的输入。样例以 Assistant 工具调用结束，表示待监督的训练目标，不是已执行完毕的旅行任务。

`city_code_mapping.json` 从本地原项目 `TripPilot-LoRA-Portfolio/configs/city_code_mapping.json` 摘取嘉兴、杭州、上海三项，只保留转换器使用的 `name` 字段。原文件有 361 项，SHA-256 为：

```text
78c38eb41a0f50f83ad575d65a064739ae38f64c774b40aac25fb1fa95093335
```

该哈希用于标识原始映射文件，不是本目录精简版的哈希；精简版校验以仓库根目录 `MANIFEST.sha256` 为准。

## 运行步骤

只需 Python 3.9+ 标准库，不需要模型权重、GPU、API Key 或网络。从仓库根目录执行（以下输出路径适用于 macOS/Linux；其他系统可换成可写路径）：

```bash
python3 03_核心代码/canonicalize_sequential_dataset.py \
  --input 03_核心代码/examples/synthetic_conversations.json \
  --mapping 03_核心代码/examples/city_code_mapping.json \
  --output /tmp/trippilot-canonicalized.json \
  --training
```

命令会写入或覆盖指定的输出文件，并打印：

```json
{"input_rows": 2, "output_rows": 3}
```

生成的三条训练样本分别以 `search_travel_guide`、`get_weather_info`、`query_route` 为最后一个 Assistant 工具目标：

- 攻略参数变为 `{"location":"嘉兴","search_mode":"hybrid"}`。
- 天气目标之前已有攻略调用及包含“3天”的合成工具摘要；天气城市统一为“嘉兴”，日期和天数保持不变。
- 路线只保留 `start_location`、`end_location`，删除 `city_code`。

`--training` 会将第一条并发调用样例拆成攻略、天气两个监督前缀。转换过程生成的攻略摘要用于展示协议，不表示访问了真实攻略服务。

从仓库根目录进入核心代码目录，运行真实 CLI 回归测试：

```bash
cd 03_核心代码
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests -p test_canonicalize_sequential_dataset.py -v
```

测试使用临时目录验证输出，不会改写本目录的输入文件。
