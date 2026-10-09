from playwright.sync_api import sync_playwright

URL = "https://steam-games-recommendation-system.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=60000)
    page.wait_for_timeout(5000)

    wake_button = page.get_by_role("button", name="Yes, get this app back up!")
    if wake_button.count() > 0:
        print("App was asleep, waking it up")
        wake_button.click()
        page.wait_for_timeout(60000)
    else:
        print("App is awake")

    browser.close()