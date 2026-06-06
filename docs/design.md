# AI 英语口语陪练 — 设计文档

## 1. 系统架构

```
┌──────────┐    WebSocket    ┌──────────────┐
│  前端页面  │ ◄──────────────► │  FastAPI 服务  │
│ (HTML/JS) │                 │  (Python)     │
└──────────┘                 └──────┬───────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
    ┌─────▼─────┐          ┌───────▼───────┐          ┌──────▼──────┐
    │  ASR 模块  │          │   Coach 模块   │          │  TTS 模块   │
    │ (Whisper)  │          │  (DeepSeek)    │          │ (edge-tts)  │
    └───────────┘          └───────┬───────┘          └─────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Evaluation 模块 │
                          │  (纠错 + 评测)   │
                          └─────────────────┘
```

## 2. 模块设计

### 2.1 ASR 模块（asr/）
- **职责**：将用户语音实时转写为文本
- **技术**：faster-whisper，使用 base 或 small 模型
- **接口**：
  - `transcribe(audio_bytes: bytes) -> str`：转写音频
  - `transcribe_stream(audio_stream) -> Iterator[str]`：流式转写

### 2.2 Coach 模块（coach/）
- **职责**：管理对话上下文，生成 AI 回复
- **技术**：DeepSeek Chat API
- **接口**：
  - `generate_reply(user_text: str, scene: str, history: list) -> str`：生成回复
  - `get_system_prompt(scene: str) -> str`：获取场景 Prompt

### 2.3 TTS 模块（tts/）
- **职责**：将 AI 文本回复合成为语音
- **技术**：edge-tts（基于微软 Edge 免费 TTS 接口）
- **接口**：
  - `synthesize(text: str) -> bytes`：合成语音

### 2.4 Evaluation 模块（evaluation/）
- **职责**：语法纠错 + 发音评测 + 课后报告
- **技术**：DeepSeek API（语法纠错）+ faster-whisper 置信度（发音参考）
- **接口**：
  - `check_grammar(text: str) -> dict`：语法纠错
  - `evaluate_pronunciation(audio: bytes, reference: str) -> dict`：发音评测
  - `generate_report(session_data: dict) -> dict`：生成课后报告

### 2.5 Web 模块（web/）
- **职责**：WebSocket 服务 + HTTP API
- **技术**：FastAPI + WebSocket
- **端点**：
  - `ws /ws/chat`：实时对话 WebSocket
  - `GET /api/scenes`：获取场景列表
  - `GET /api/report/{session_id}`：获取课后报告

### 2.6 前端（frontend/）
- **职责**：用户交互界面
- **技术**：原生 HTML/CSS/JS + Web Audio API
- **功能**：场景选择、录音按钮、对话展示、字幕显示

## 3. 数据流

```
用户说话 → 麦克风采集 → VAD 切句 → ASR 转写 → 
→ Coach 生成回复 → TTS 合成语音 → 扬声器播放
→ Evaluation 记录纠错 → 课后生成报告
```

## 4. WebSocket 消息协议

### 客户端 → 服务端
```json
{"type": "audio", "data": "<base64>"}
{"type": "start_scene", "scene": "interview"}
{"type": "interrupt"}
{"type": "end_session"}
```

### 服务端 → 客户端
```json
{"type": "transcript", "text": "用户说的话"}
{"type": "reply", "text": "AI 回复文本"}
{"type": "audio", "data": "<base64>"}
{"type": "correction", "original": "...", "suggestion": "..."}
{"type": "report", "data": {...}}
```

## 5. 场景 Prompt 设计

每个场景的 System Prompt 包含：
- 角色设定（面试官 / 服务员 / 会议主持人）
- 对话风格（正式 / 随意 / 商务）
- 难度级别（初级 / 中级 / 高级）
- 纠错策略（立即纠正 / 延迟纠正）

## 6. 错误修正策略

| 错误类型 | 严重级别 | 修正时机 | 修正方式 |
|----------|:--------:|----------|----------|
| 严重影响理解 | 🔴 致命 | 立即 | 打断对话，提示正确表达 |
| 语法小错 | 🟡 轻微 | 课后 | 报告中列出 |
| 发音不准 | 🟡 轻微 | 课后 | 报告中标注 |
| 用词不当 | 🟢 建议 | 课后 | 报告中提供替代词汇 |
