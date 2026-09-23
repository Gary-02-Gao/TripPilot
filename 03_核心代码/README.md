# 核心代码

## 本地运行

在本目录执行（Python 3.9+ 标准库）：

```bash
python3 demo_server.py --host 127.0.0.1 --port 8091
```

打开 <http://127.0.0.1:8091>，查看旅行工作台、工具轨迹、LoRA 归档样本及评测页。演示包含嘉兴、杭州、上海三个城市的模型生成攻略样例。

普通对话由离线控制器和模拟工具驱动；归档入口加载真实模型评测记录。服务采用本地单用户会话，默认绑定回环地址。

## 实现索引

| 文件 | 审阅重点 |
| --- | --- |
| `canonicalize_sequential_dataset.py` | 城市/参数规范化、依赖式调用拆分 |
| `conversation_splitter.py` | 多轮样本构造与上下文保留 |
| `build_weighted_repair_dataset.py` | 困难类型重采样与修复数据构建 |
| `inspect_qwen_dataset.py` | Assistant 区间定位与 Loss Mask |
| `train_qwen_last_assistant_lora.py` | 最后一条 Assistant 监督、LoRA、BF16/DDP |
| `merge_lora_into_base.py` | 按记录顺序合并适配器 |
| `evaluate_qwen_fc.py` | Gold 历史逐节点评测、JSON 解析、严格参数匹配 |
| `merge_eval_shards.py` | 分片计数汇总 |
| `all_tools_sequential_v2.json` | 五个工具的参数契约 |
| `demo_agent.py`、`demo_server.py`、`web/` | 离线产品流程、服务端输入校验、前端展示 |

## 离线测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

**15/15 通过。** 覆盖工具依赖顺序、需求补全、酒店链、路线参数契约、领域拒答、参数规范化与 Function Calling 解析；包含异常模型 JSON、混合损坏调用和实际数据转换 CLI 的回归测试。原始浏览器检查记录见 `../02_成果与验证/frontend-verification.json`，最新核验见 `../本地环境验证.md`。

数据转换所需的最小城市映射与合成输入已放在 [examples/](examples/README.md)，可以完全使用仓内文件验证顺序化转换；它们不替代原始训练数据。

## 训练与模型重建

训练代码使用 PyTorch、Transformers 与 PEFT，运行时需准备兼容环境、匹配的模型权重及数据。适配器按 `../02_成果与验证/model-chain.json` 记录的三阶段顺序合并，最终模型由完整合并链重建。

本仓库提供精选代码与评测证据，不包含完整训练/测试数据或模型权重。运行评测时需显式指定 `--data-file` 与 `--tools-file all_tools_sequential_v2.json`；数据检查脚本对应参数为 `--data_file` 与 `--tools_file all_tools_sequential_v2.json`。

源码与来源哈希见 `../02_成果与验证/source-manifest.json`。本目录按数据处理、训练、评测与前端流程组织精选实现。
