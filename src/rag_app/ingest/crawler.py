"""Crawl a JavaScript-rendered, paginated listing page with Playwright.

Plain ``httpx`` + ``BeautifulSoup`` cannot see links that are injected by
client-side JavaScript or revealed by lazy loading / pagination. This module
drives a real Chromium browser (Playwright sync API) to render the page, scroll
it, click through pagination, and collect the links that matter.
"""
from pathlib import Path
from typing import Callable

from rag_app.logging_utils import get_logger

log = get_logger(__name__)

# Selectors that commonly dismiss a cookie / consent banner. Tried in order;
# the first one that is visible is clicked.
_COOKIE_SELECTORS = [
    "#awsccc-cb-btn-accept",                       # AWS consent banner "Accept all"
    "button[data-id='awsccc-cb-btn-accept']",
    "button:has-text('Accept all')",
    "button:has-text('Accept cookies')",
    "button:has-text('Accept')",
    "#onetrust-accept-btn-handler",
]

# Selectors that advance to the next page / load more results. Tried in order;
# the first one that is visible and enabled is clicked. The AWS whitepapers grid
# uses a "Show N more" button (e.g. "Show 8 more") that eventually turns into
# "Show less" — ``_click_next`` skips any control whose label says "less".
_NEXT_SELECTORS = [
    "button[aria-label='Next page']",
    "a[aria-label='Next page']",
    "button[aria-label*='Next']",
    "a[aria-label*='Next']",
    "button:has-text('Show more')",
    "button:has-text('Show')",      # AWS "Show 8 more" / "Show 2 more"
    "button:has-text('Load more')",
    "button:has-text('View more')",
    "button:has-text('more')",
    ".m-icon-angle-right",          # AWS directory "next" chevron
    "li.m-page-next > a",
    "a[rel='next']",
    "button[rel='next']",
]

# How many consecutive pages with no new links to tolerate before stopping. A
# "load more" grid can briefly return nothing while the next batch streams in, so
# we do not stop on the first empty page.
_MAX_STALE_PAGES = 5


def _accept_cookies(page) -> None:
    for selector in _COOKIE_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                el.click()
                page.wait_for_timeout(500)
                log.info("Accepted cookie banner via %r", selector)
                return
        except Exception:  # noqa: BLE001 - banner handling is best-effort
            continue


def _scroll_to_bottom(page, steps: int = 12, pause_ms: int = 400) -> None:
    """Scroll down in steps to trigger lazy loading."""
    for _ in range(steps):
        page.mouse.wheel(0, 20000)
        page.wait_for_timeout(pause_ms)
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(pause_ms)


def _collect_links(page, is_relevant: Callable[[str], bool]) -> list[str]:
    hrefs = page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.href)"
    )
    return [h for h in hrefs if is_relevant(h)]


def _click_next(page) -> bool:
    """Click the first visible, enabled pagination control. Return True if clicked."""
    for selector in _NEXT_SELECTORS:
        try:
            el = page.query_selector(selector)
            if not (el and el.is_visible() and el.is_enabled()):
                continue
            label = (el.text_content() or "") + (el.get_attribute("aria-label") or "")
            if "less" in label.lower():  # skip a "Show less" collapse control
                continue
            el.scroll_into_view_if_needed()
            el.click()
            page.wait_for_timeout(1500)
            log.info("Clicked pagination control %r", selector)
            return True
        except Exception:  # noqa: BLE001 - try the next fallback selector
            continue
    return False


def crawl_links(
    start_url: str,
    is_relevant: Callable[[str], bool],
    max_pages: int = 50,
    headless: bool = True,
    debug_screenshot: str | Path = "crawl_debug.png",
) -> list[str]:
    """Open ``start_url`` in Chromium and collect relevant links across pages.

    Scrolls each page to trigger lazy loading, keeps links where
    ``is_relevant(link)`` is True, then clicks the next-page / load-more control
    and repeats until no new links appear or ``max_pages`` is reached.

    If zero links are found, saves a full-page screenshot to ``debug_screenshot``
    and logs a warning so the selectors can be inspected.
    """
    from playwright.sync_api import sync_playwright

    found: list[str] = []
    seen: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(2000)
            _accept_cookies(page)

            stale = 0
            for page_num in range(1, max_pages + 1):
                _scroll_to_bottom(page)
                new = [l for l in _collect_links(page, is_relevant) if l not in seen]
                for link in new:
                    seen.add(link)
                    found.append(link)
                log.info("Page %d: +%d new links (%d total)", page_num, len(new), len(found))

                stale = 0 if new else stale + 1
                if stale >= _MAX_STALE_PAGES:
                    log.info("No new links for %d pages; stopping.", stale)
                    break

                if not _click_next(page):
                    log.info("No further pagination control found; stopping.")
                    break

            if not found:
                shot = Path(debug_screenshot)
                try:
                    page.screenshot(path=str(shot), full_page=True)
                    log.warning(
                        "Found 0 links. Saved debug screenshot to %s", shot.resolve()
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Found 0 links and failed to save screenshot: %s", exc)
        finally:
            browser.close()

    return found
