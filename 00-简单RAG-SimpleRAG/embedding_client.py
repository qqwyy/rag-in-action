# embedding_client.py

import requests
import json
from typing import Optional, Dict, Any

def get_embedding_response(
    model: str,
    input_text: str | list[str],
    base_url: str,
    auth_key: str
) -> Dict[Any, Any]:
    # print(f"入参：{model} ，{input_text} ，{base_url}，{auth_key}")
    url = base_url
    headers = {"Content-Type": "application/json"}
    if auth_key:
        headers["Authorization"] = auth_key

    payload = {
        "model": model,
        "input": input_text
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise ValueError(
            f"API 请求失败 [HTTP {response.status_code}]: {response.text}"
        )

    return response.json()


def extract_embeddings_from_response(response: Dict[Any, Any]) -> list[list[float]]:
    """
    从 embedding API 响应中提取向量列表。

    Args:
        response (dict): 由 get_embedding_response 返回的响应

    Returns:
        list[list[float]]: 向量列表。即使输入是单个字符串，也返回 [[vec]]
    """
    data = response.get("data", [])
    return [item["embedding"] for item in data]