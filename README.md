# AI 英语口语陪练（AI English Speaking Coach）

基于 AI 的英语口语练习工具，支持多场景沉浸式对话训练。

## 功能

- 场景对话：面试、点餐、商务会议等真实场景模拟
- 语音交互：实时语音识别与合成，自然对话体验
- 语法纠错：实时检测并纠正表达错误
- 发音评测：对发音准确性进行评分
- 课后报告：自动生成多维度学习报告

## 技术栈

| 模块 | 技术 |
|------|------|
| 语音识别 | faster-whisper（本地 GPU 推理） |
| 对话引擎 | DeepSeek Chat API |
| 语音合成 | edge-tts（免费） |
| 后端服务 | FastAPI + WebSocket |
| GPU 加速 | CUDA 12.6 + cuDNN 9.x |

## 快速开始

### 环境要求

- Python 3.11
- NVIDIA GPU（CUDA 12.6）
- 麦克风 + 扬声器

### 安装

```bash
pip install -r requirements.txt
```

### 配置

在 `.env` 文件中设置 API Key：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 运行

```bash
python main.py
```

浏览器打开 `http://localhost:8000` 即可开始使用。

## 项目结构

```
MyDemo/
├── main.py              # 入口
├── config.py            # 配置管理
├── asr/                 # 语音识别
├── tts/                 # 语音合成
├── coach/               # 对话引擎
├── evaluation/          # 纠错 + 评测 + 报告
├── web/                 # FastAPI 服务
├── frontend/            # 前端页面
├── scenes/              # 场景 Prompt 模板
└── docs/                # 项目文档
```

## 开发计划

详见 [docs/project_plan.md](docs/project_plan.md)