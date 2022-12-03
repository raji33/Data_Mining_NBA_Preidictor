def scrape_game(standings_file):
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if
                  l and "boxscore" in l and '.html' in l]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+") as f:
            f.write(html)


# Make sure to install playwright browsers by running playwright install on the command line or !playwright install from Jupyter
def get_html(url, selector, sleep=15, retries=3):
    html = None
    for i in range(1, retries + 1):
        time.sleep(sleep * i)
        try:
            with sync_playwright() as p:
                browser = p.firefox.launch()
                page = browser.new_page()
                page.goto(url)
                print(page.title())
                html = page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html


def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = get_html(url, "#content .filter")

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)


if __name__ == '__main__':

    import os

    SEASONS = list(range(2017, 2024))

    DATA_DIR = "data"
    STANDINGS_DIR = os.path.join(DATA_DIR, "standings1")
    SCORES_DIR = os.path.join(DATA_DIR, "scores1")

    from bs4 import BeautifulSoup

    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    import time

    for season in SEASONS:
        scrape_season(season)

    standings_files = os.listdir(STANDINGS_DIR)

    import pandas as pd

    for season in SEASONS:
        files = [s for s in standings_files if str(season) in s]

        for f in files:
            filepath = os.path.join(STANDINGS_DIR, f)

            scrape_game(filepath)

    # import os
    # SEASONS = list(range(2017,2024))
    # DATA_DIR = "data"
    # STANDINGS_DIR = os.path.join(DATA_DIR, "standings1")
    # SCORES_DIR = os.path.join(DATA_DIR, "scores1")
    #
    # from bs4 import BeautifulSoup
    # from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    # import time
    #
    # #comment out because we already scraped each seasing
    # for season in SEASONS:
    #     print(season)
    #     scrape_season(season)
    #
    # standings_files = os.listdir(STANDINGS_DIR)
    # # standings_file = os.path.join(STANDINGS_DIR, standings_files[0])
    # # with open(standings_file, 'r') as f:
    # #     html = f.read()
    # #
    # # soup = BeautifulSoup(html,features="html.parser" )
    # # links = soup.find_all("a")
    # # hrefs = [l.get("href") for l in links]
    # # box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    # # box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]
    # #
    # #
    # # for url in box_scores:
    # #     save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
    # #     print(save_path)
    # #     if os.path.exists(save_path):
    # #         continue
    # #     html = get_html(url, "#content")
    # #     if not html:
    # #         continue
    # #     # with open(save_path, "w+") as f:
    # #     #     f.write(html)
    #
    #
    #
    # for f in standings_files:
    #     filepath = os.path.join(STANDINGS_DIR, f)
    #     scrape_game(filepath)
