"""可选脚本：用 Playwright 抓取真实运行界面的截图（README 用图）。

前置：
    1. 先启动服务：``uvicorn app.main:app --port 8000``
    2. 安装截图工具（不属于项目运行时依赖）：
       ``py -3.14 -m pip install playwright && py -3.14 -m playwright install chromium``

用法：
    ``py -3.14 -m scripts.capture_screenshots``

脚本只做“打开页面 → 输入问题 → 截图”，不修改任何数据；
图片输出到 ``docs/screenshots/``。
"""

from __future__ import annotations

from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"

QUESTION = "报销制度规定一线城市住宿标准为每晚多少元？"


def capture() -> None:
    from playwright.sync_api import sync_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1440, "height": 940},
            device_scale_factor=2,
            locale="zh-CN",
        )
        page = context.new_page()
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_selector("#doc-list .doc-item", timeout=20_000)

        page.screenshot(path=OUTPUT_DIR / "01-overview.png", full_page=True)
        print("已保存 01-overview.png")

        page.select_option("#top-k", "3")
        page.fill("#question", QUESTION)
        page.click("#send")
        page.wait_for_selector(".msg-assistant .answer-text", timeout=60_000)
        page.wait_for_timeout(500)

        # 让答案开头出现在可视区顶部
        page.eval_on_selector(".msg-assistant", "el => el.scrollIntoView({ block: 'start' })")
        page.wait_for_timeout(300)
        page.screenshot(path=OUTPUT_DIR / "02-answer-dark.png", full_page=True)
        print("已保存 02-answer-dark.png")

        # 展开第一条引用 + 检索调试，切浅色主题后截取底部
        page.click("[data-toggle='1']")
        page.click("details.debug summary")
        page.click("#theme-toggle")
        page.wait_for_timeout(400)
        page.eval_on_selector("details.debug", "el => el.scrollIntoView({ block: 'end' })")
        page.wait_for_timeout(300)
        page.screenshot(path=OUTPUT_DIR / "03-debug-light.png", full_page=True)
        print("已保存 03-debug-light.png")

        # 窄屏（移动端布局）
        page.set_viewport_size({"width": 430, "height": 900})
        page.wait_for_timeout(500)
        page.eval_on_selector("#messages", "el => (el.scrollTop = 0)")
        page.screenshot(path=OUTPUT_DIR / "04-mobile.png", full_page=True)
        print("已保存 04-mobile.png")

        context.close()
        browser.close()


if __name__ == "__main__":
    capture()
