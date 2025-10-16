# Meeting Planner Agent

一个能自动完成会议安排全流程的智能 Agent，支持：
- 自然语言时间解析（“下周二下午2点”）
- 多人日历空闲检查
- 冲突检测与备选时间建议
- 自动创建日历事件 + 发送邮件
- 用户确认交互闭环

## 技术亮点
- **三层 Prompt 设计**：意图提取 → 任务规划 → 用户协商
- **真实 API 集成**：Google Calendar + Gmail（支持 mock 模式）
- **错误恢复**：冲突时提供备选方案
- **工程化结构**：工具解耦、配置分离、状态清晰

## 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. （可选）配置 Google Cloud API，下载 `credentials.json`
3. 设置环境变量（或修改 config.py）：
   ```bash
   export ORGANIZER_EMAIL=your.email@example.com
   export USE_MOCK=True  # 设为 True 跳过 Google API
