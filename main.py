import json
from agent.prompt_templates import INTENT_EXTRACTION_PROMPT
from agent.planner import plan_and_execute_meeting
from langchain_community.chat_models import ChatTongyi
import os

os.environ["DASHSCOPE_API_KEY"] = "your-api-key" 

def extract_intent(user_input: str):
    llm = ChatTongyi(model_name="qwen-max")
    chain = INTENT_EXTRACTION_PROMPT | llm
    response = chain.invoke({"input": user_input})
    try:
        return json.loads(response.content)
    except:
        raise ValueError("无法解析用户意图，请提供更清晰的请求。")

def main():
    user_input = input("请输入会议安排请求：")
    print("正在解析意图...")
    
    intent = extract_intent(user_input)
    print("提取的意图：", json.dumps(intent, indent=2, ensure_ascii=False))
    
    print("正在规划会议...")
    result = plan_and_execute_meeting(intent)
    
    if result["status"] == "needs_confirmation":
        alt = result["alternative"]
        print(f"\n⚠️ 时间冲突！建议改为：{alt[0].strftime('%Y-%m-%d %H:%M')} - {alt[1].strftime('%H:%M')}")
        confirm = input("是否接受？(y/n): ")
        if confirm.lower() == "y":
            # 这里可继续调用 create_event 和 send_email
            print("会议已按新时间安排！")
        else:
            print("已取消。")
    else:
        print("✅ 会议安排成功！详情：", result["event"].get("htmlLink", "无链接"))

if __name__ == "__main__":
    main()
