# AI 英语口语陪练 — 项目阶段控制

## 阶段总览

| 阶段 | 名称 | 目标 | 预计耗时 | 状态 |
|------|------|------|:--:|:--:|
| P0 | 项目初始化 | 文档 + 骨架 + 环境配置 | 0.5 天 | ✅ 已完成 |
| P1 | 对话引擎 | DeepSeek 集成 + 场景 Prompt | 1 天 | ✅ 已完成 |
| P2 | 语音识别 | Whisper 接入 + 音频采集 | 1 天 | ✅ 已完成 |
| P3 | 语音合成 | edge-tts 接入 + 音频播放 | 0.5 天 | ✅ 已完成 |
| P4 | 服务串联 | FastAPI + WebSocket + 前端 | 1.5 天 | ✅ 已完成 |
| P5 | 纠错评测 | 语法纠错 + 发音评测 | 1 天 | ✅ 已完成 |
| P6 | 课后报告 | 总结报告生成 + 展示 | 0.5 天 | ⬜ |
| P7 | 打磨上线 | 延迟优化 + 打断机制 + 测试 | 1 天 | ⬜ |

**总预计：7 天**

## P0：项目初始化 ✅

### 交付物
- [x] 需求文档
- [x] 设计文档
- [x] 项目计划
- [x] README
- [x] 项目骨架（目录 + 空模块）
- [x] 环境配置（.env + requirements.txt）
- [x] 基础 config.py

### 完成标准
- [x] 所有模块 __init__.py 就位
- [x] config.py 可正确加载 API Key
- [x] requirements.txt 可一键安装所有依赖

## P1：对话引擎 ✅

### 交付物
- [x] coach/scene_manager.py：场景管理（加载/切换/热重载 YAML 模板）
- [x] coach/engine.py：DeepSeek API 异步对话引擎（支持流式 SSE 兼容输出、异常重试）
- [x] scenes/interview.yaml：面试场景
- [x] scenes/ordering.yaml：点餐场景
- [x] scenes/meeting.yaml：会议场景
- [x] scenes/free_talk.yaml：自由对话场景
- [x] scenes/travel.yaml：旅行场景
- [x] coach/tests/：24 个单元测试（场景加载/切换/热重载、历史管理/消息构建），全部通过

### 完成标准
- [x] 命令行可输入文本，获得对应场景的 AI 回复
- [x] 回复内容与场景高度相关
- [x] 支持多轮对话上下文

## P2：语音识别 ✅

### 交付物
- [x] asr/recognizer.py：Whisper 识别封装（faster-whisper GPU 推理，整段转写 + 流式转写，延迟加载）
- [x] asr/audio_capture.py：麦克风音频采集（pyaudio 回调模式 + queue.Queue，暂停/恢复，上下文管理器）
- [x] asr/vad.py：语音活动检测（webrtcvad，帧级检测 + 语音段切分）
- [x] asr/__init__.py：模块导出
- [x] asr/tests/test_vad.py：VAD 单元测试（静默/混合音频、帧长、segment 切分）
- [x] asr/tests/test_audio_capture.py：AudioCapture 配置与队列逻辑测试
- [x] asr/tests/test_recognizer.py：WhisperRecognizer 初始化与配置读取测试
- [x] config.py：新增 WHISPER_MODEL_SIZE、SAMPLE_RATE 配置项
- [x] requirements.txt：补充 pyaudio 依赖

### 完成标准
- [x] 从麦克风录音，实时转写为文本
- [x] 安静环境识别准确率 ≥ 90%
- [x] GPU 推理延迟 ≤ 500ms

## P3：语音合成 ✅

### 交付物
- [x] tts/synthesizer.py：edge-tts 封装（文件合成 + 流式合成 + 语音列表查询）
- [x] tts/__init__.py：模块导出（Synthesizer, text_to_speech）
- [x] tts/tests/test_synthesizer.py：Synthesizer 单元测试（初始化/验证/异常处理/流式合成/list_voices）
- [x] config.py TTS 配置项：TTS_VOICE、TTS_RATE、TTS_PITCH
- [x] requirements.txt 依赖补充（edge-tts 已有）

### 完成标准
- [x] 文本合成语音输出为音频文件/流
- [x] 语音自然度良好
- [x] 合成延迟 ≤ 1 秒

## P4：服务串联

### 交付物
- web/server.py：FastAPI + WebSocket 服务
- web/handler.py：消息处理与模块调度
- frontend/index.html：对话界面

### 完成标准
- 浏览器打开页面，可选择场景
- 点击录音按钮开始对话
- 听到 AI 语音回复
- 看到对话文本记录

## P5：纠错评测 ✅

### 交付物
- [x] evaluation/__init__.py — 模块导出（GrammarChecker, PronunciationEvaluator）
- [x] evaluation/grammar_checker.py — 基于 DeepSeek API 的语法纠错（语法/词汇/风格三类错误检测，结构化 JSON 输出）
- [x] evaluation/pronunciation.py — 基于 Whisper 转写 + WER 文本相似度的发音评测
- [x] evaluation/tests/__init__.py
- [x] evaluation/tests/test_grammar_checker.py — 12 个单元测试（正常纠错/无错/多错/API异常/Markdown包裹/空响应/字段缺失等）
- [x] evaluation/tests/test_pronunciation.py — 15 个单元测试（完全匹配/部分匹配/无匹配/空参考/文件缺失/转写异常/WER计算/文本归一化等）
- [x] config.py — 追加 GRAMMAR_CHECK_MODEL、PRONUNCIATION_THRESHOLD 配置项
- [x] web/handler.py — MessageHandler 新增 check_grammar()、evaluate_pronunciation() 方法
- [x] web/server.py — WebSocket 新增 "check_grammar" 和 "evaluate_pronunciation" 消息类型处理

### 完成标准
- [x] 语法错误被检测并给出建议（grammar/vocabulary/style 三维度）
- [x] 发音评测给出评分（overall_score / fluency / accuracy / 逐词详情）
- [x] API 不可用时优雅降级，不抛异常
- [x] 音频文件不存在/静音/转写失败等边界情况妥善处理

## P6：课后报告

### 交付物
- evaluation/report.py：报告生成
- frontend 报告展示页

### 完成标准
- 对话结束后自动生成报告
- 包含多维度评分和错误列表

## P7：打磨上线

### 交付物
- 打断机制实现
- 延迟优化
- 异常处理完善

### 完成标准
- 端到端延迟 ≤ 2 秒
- 打断响应 ≤ 500ms
- 无明显 Bug
