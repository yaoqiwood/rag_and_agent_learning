import os

import requests


def set_proxy():
    # 统一使用 http 协议头，即使是 https_proxy
    proxy_url = "http://127.0.0.1:7890"

    # 1. 尝试环境变量
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url

    # 2. 显式在 requests 中指定代理（这是最稳妥的测试方法）
    proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }

    url = "https://www.google.com"  # 换成 google 测试更准
    try:
        # 显式传入 proxies 参数
        response = requests.get(url, timeout=5, proxies=proxies)
        print(f"显式代理测试响应头: {response}")
        if response.status_code == 200:
            print("✅ 显式代理测试成功！")
        else:
            print(f"⚠️ 状态码异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 还是不通，错误详情: {e}")
