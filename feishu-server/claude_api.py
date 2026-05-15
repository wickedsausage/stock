import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# 从项目 CLAUDE.md 读取上下文
def _load_project_context():
    import os
    claude_md = os.path.join(os.path.dirname(__file__), "..", ".claude", "CLAUDE.md")
    try:
        with open(claude_md, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


SYSTEM_PROMPT = f"""你是 Claude，一个在 Windows 11 上运行的 AI 编程助手。你在帮助一位从事量化学习项目的用户。

当前项目路径: C:\\Users\\社畜的肯定\\Desktop\\量化学习

项目上下文:
{_load_project_context()}

回复风格：
- 用中文回复
- 简洁直接
- 代码相关问题时给出具体路径和行号
- 语气友好但不啰嗦"""


class ClaudeClient:
    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("CLAUDE_MODEL", "claude-opus-4-7")

    def chat(self, message):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
            )
            reply = response.content[0].text
            logger.info(f"Claude 回复: {reply[:100]}...")
            return reply
        except Exception as e:
            logger.error(f"Claude API 调用失败: {e}")
            return f"抱歉，AI 服务出错了：{e}"
