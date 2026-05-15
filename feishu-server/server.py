import json
import logging
from flask import Flask, request, jsonify
from feishu_api import FeishuClient
from claude_api import ClaudeClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
feishu = FeishuClient()
claude = ClaudeClient()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/feishu/callback', methods=['POST'])
def feishu_callback():
    data = request.get_json()
    logger.info(f"收到飞书回调: {json.dumps(data, ensure_ascii=False)}")

    # URL 验证 (challenge)
    if data.get("type") == "url_verification":
        challenge = data.get("challenge", "")
        logger.info(f"URL 验证, challenge: {challenge}")
        return jsonify({"challenge": challenge})

    # 消息事件
    header = data.get("header", {})
    event = data.get("event", {})
    event_type = header.get("event_type", "")

    if event_type == "im.message.receive_v1":
        message = event.get("message", {})
        msg_type = message.get("message_type", "")
        content_str = message.get("content", "{}")

        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {}

        text = content.get("text", "") or ""
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})
        open_id = sender_id.get("open_id", "")

        logger.info(f"收到消息: open_id={open_id}, text={text}")

        if text and open_id:
            # 调用 Claude API 生成回复
            reply = claude.chat(text)
            # 发送回复
            feishu.send_text_message(open_id, reply)

    return jsonify({"code": 0})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
