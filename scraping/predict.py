def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

# this will pull won column from next game and pull it back one row
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group


def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)



if __name__ == '__main__':
    import pandas as pd
    pd.options.mode.chained_assignment = None

    df = pd.read_csv("nba_games.csv", index_col=0)

    df = df.sort_values("date")

    df = df.reset_index(drop=True)

    del df["mp.1"]
    del df["mp_opp.1"]
    del df["index_opp"]

    df = df.groupby("team", group_keys=False).apply(add_target)

    df["target"][pd.isnull(df["target"])] = 2
    df["target"] = df["target"].astype(int, errors="ignore")

    nulls = pd.isnull(df).sum()
    nulls = nulls[nulls > 0]
    valid_columns = df.columns[~df.columns.isin(nulls.index)]
    df = df[valid_columns].copy()

    from sklearn.linear_model import RidgeClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import TimeSeriesSplit

    rr = RidgeClassifier(alpha=1)

    split = TimeSeriesSplit(n_splits=3)

    sfs = SequentialFeatureSelector(rr,n_features_to_select=30,direction="forward",cv=split, n_jobs=1)

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    sfs.fit(df[selected_columns], df["target"])

    SequentialFeatureSelector(cv=TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None),
                              estimator=RidgeClassifier(alpha=1),
                              n_features_to_select=30, n_jobs=1)

    predictors = list(selected_columns[sfs.get_support()])

    predictions = backtest(df, rr, predictors)



    #the predictions tells us the actual and prediction for each of the columns
    #we dont want to look through the entire table so we use metric acturacy score


    from sklearn.metrics import accuracy_score

    # gives accuracy prediction was right in percentage
    print(accuracy_score(predictions["actual"], predictions["prediction"]))

    #group acuraccy by the home column and apply function what % of the time the team wins based on wether they are home or not
    df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])


    df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

    #groups row and the previous 10 rows and get the mean -- gets the avg preformance of the team that played from their last 10 games
    #only use rolling averages from that season bc teams can change drastically from season to season
    df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    #adds _10 to each column name indicateing the rolling avg
    rolling_cols = [f"{col}_10" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols
    df = pd.concat([df, df_rolling], axis=1)

    #drops the rows that have missing values -- drops 1st 10 games in season bc they dont have rolling avg of the previous 10 games
    df = df.dropna()

    #give algo information we know certain information before the game is played
    #give our algo knowledge about the team to help with predictions
    #know infromation about team playing and if home or not and the date of the game
    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"],
                    right_on=["team_opp_next", "date_next"])

    full[["team_x", "team_opp_next_x", "team_y", "team_opp_next_y", "date_next"]]

    #columns we will not pass into ml model
    removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

    #columns we are going to be putting into model
    selected_columns = full.columns[~full.columns.isin(removed_columns)]
    sfs.fit(full[selected_columns], full["target"])

    #these are the selected columns from the feature selector
    predictors = list(selected_columns[sfs.get_support()])

    predictions = backtest(full, rr, predictors)

    print(accuracy_score(predictions["actual"], predictions["prediction"]))

    #we used a ridge regression model but we could try to use more powerful model to improve accuracy
    #use xgboost or randomforestclassifier instead of ridgeClassifier
    #try changing the number of features for the model or try backward feature selection
    #try different ratios --  instead of last 10 games do last 5 games or last 20 games


    #predicting future rows
    #need to modify code to get most up to date box scores.
    #re run the model and manually or programatically fill in the (home_next, team_opp_next date_next - for both team and opponent and merge it in dataset
    #re run predictions and look at any rows where target is 2 -- when game is in future and find prediction in that
