import json
import time
import logging
import requests

logger = logging.getLogger(__name__)

FEISHU_BASE = "https://open.feishu.cn/open-apis"


class FeishuClient:
    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.app_id = os.getenv("FEISHU_APP_ID", "")
        self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self._token = None
        self._token_expire = 0

    def _get_tenant_access_token(self):
        if self._token and time.time() < self._token_expire - 60:
            return self._token

        resp = requests.post(
            f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise Exception(f"获取飞书 token 失败: {data}")

        self._token = data["tenant_access_token"]
        self._token_expire = time.time() + data.get("expire", 7200)
        logger.info("飞书 tenant_access_token 已刷新")
        return self._token

    def send_text_message(self, open_id, text):
        token = self._get_tenant_access_token()
        resp = requests.post(
            f"{FEISHU_BASE}/im/v1/messages?receive_id_type=open_id",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "receive_id": open_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.error(f"发送飞书消息失败: {data}")
            return False
        logger.info(f"消息已发送到 open_id={open_id}")
        return True
