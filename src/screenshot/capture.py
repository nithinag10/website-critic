import asyncio
from playwright.async_api import async_playwright

async def auto_scroll(page):
    """Scrolls down the page gradually to trigger lazy-loading."""
    await page.evaluate("""
        async () => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                const distance = 100;
                const timer = setInterval(() => {
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    if (totalHeight >= document.body.scrollHeight) {
                        clearInterval(timer);
                        resolve();
                    }
                }, 100);
            });
        }
    """)

async def capture_screenshot(url: str) -> bytes:
    """Captures full page screenshot."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            device_scale_factor=2
        )
        page = await context.new_page()
        
        page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(2000)
        await auto_scroll(page)
        await page.wait_for_timeout(2000)
        
        screenshot_bytes = await page.screenshot(full_page=True, timeout=60000)
        await browser.close()
        return screenshot_bytes