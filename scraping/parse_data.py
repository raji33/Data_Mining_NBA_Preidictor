def parse_html(box_score):
    #takes in path to box score file
    with open(box_score, encoding='utf-8', errors='ignore') as f:
        html = f.read()

    soup = BeautifulSoup(html, features="html.parser")
    #selects the tr element with over_header class
    #decompose removes all these elements from html
    #decompose runs in place which chaanges actual data not copy of data
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_line_score(soup):
    #gets html as a string and tell panadas to process the table that has the id "line_score"
    line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score


#grab 2 tables per team - basic stats and advanced stats for each team
def read_stats(soup, team, stat):
    df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col = 0)[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season


if __name__ == '__main__':
    import os
    import pandas as pd
    from bs4 import BeautifulSoup

    #define score directory
    SCORE_DIR = "data/scores"
    box_scores = os.listdir(SCORE_DIR)

    box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores]
    #131 is an issue (2016 04010MEM , MIL, NYK
    # box_score_2000 = box_scores[1000:2000]
    # b = [x for i, x in enumerate(box_score_2000) if i != 131 and i != 132 and i != 133]
    base_cols = None
    games = []


    for box_score in box_scores:
        check_file = os.stat(box_score).st_size
        if (check_file == 0):
            print(box_score)
            continue
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        teams = list(line_score["team"])

        summaries = []
        for team in teams:
            basic = read_stats(soup,team,"basic")
            advanced = read_stats(soup,team, "advanced")

            #grabs total for both basic and advanced stats of all players on team
            totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
            totals.index = totals.index.str.lower()

            #grabs the height value for each category done by any player on team
            maxes = pd.concat([basic.iloc[:-1,:].max(), advanced.iloc[:-1,:].max()])
            maxes.index = maxes.index.str.lower() + "_max"

            #combies basic and advanced stats into one column
            summary = pd.concat([totals,maxes])

            if base_cols is None:
                base_cols = list(summary.index.drop_duplicates(keep="first"))
                base_cols = [b for b in base_cols if "bpm" not in b]

            summary = summary[base_cols]

            summaries.append(summary)

        summary = pd.concat(summaries, axis=1).T

        game = pd.concat([summary,line_score], axis=1)

        #1st team is always away and second is home- 0 == away.
        game["home"] = [0,1]

        #concatinate opponenet states with team that played to use in ML model
        #reverse dataframe so 1st row in now second row -- home team is first
        game_opp = game.iloc[::-1].reset_index()
        game_opp.columns +="_opp"

        #gives both team states and opponent states for each team
        full_game = pd.concat([game,game_opp] ,axis=1)

        full_game["season"] = read_season_info(soup)

        full_game["date"] = os.path.basename(box_score)[:8]

        full_game["date"] = pd.to_datetime(full_game["date"] , format="%Y%m%d")

        full_game["won"] = full_game["total"] > full_game["total_opp"]
        games.append(full_game)

        if len(games) % 100 == 0:
            print(f"{len(games)} / {len(box_scores)}")

    games_df = pd.concat(games, ignore_index=True)
    games_df.to_csv("nba_games.csv")

