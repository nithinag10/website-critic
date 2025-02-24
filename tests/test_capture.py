import pytest
import asyncio
from src.screenshot.capture import capture_screenshot, auto_scroll

@pytest.mark.asyncio
async def test_capture_screenshot():
    url = "https://example.com"
    screenshot_bytes = await capture_screenshot(url)
    assert isinstance(screenshot_bytes, bytes)
    assert len(screenshot_bytes) > 0

@pytest.mark.asyncio
async def test_auto_scroll():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://example.com")
        await auto_scroll(page)
        height = await page.evaluate("document.body.scrollHeight")
        assert height > 0