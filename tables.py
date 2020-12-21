from gccr002 import *
SAVE = True
CONTEXT = None


def save_table(df, name):
    if True:
        path = pathlib.Path('figures') / 'tables'
        if isinstance(CONTEXT, int):
            path = path / ('Table %d' % CONTEXT)
        elif isinstance(CONTEXT, str):
            if CONTEXT.isnumeric():
                path = path / ('Table %s' % CONTEXT)
            elif CONTEXT[0]=='S' and CONTEXT[1:].isnumeric():
                path = path / ('Table %s' % CONTEXT)
            else:
                path = path / CONTEXT
        path.mkdir(exist_ok=True, parents=True)
        try:
            df.to_csv(path / ('%s.csv' % name))
        except:
            df.data.to_csv(path / ('%s.csv' % name))


def auto_save_table(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if SAVE:
            jsonable = [x for x in args if isinstance(x, (int, str))]
            jsonable += [x for x in kwargs.values() if isinstance(x, (int, str))]
            h = hashlib.sha1(json.dumps(jsonable).encode("utf-8")).hexdigest()[:10]
            save_table(df, '%s_%s' % (func.__name__, h))
        return df
    return wrapper


@auto_save_table
def lab_vs_clinical(df, diagnoses, senses):
    features = ["%s_change_illness" % sense for sense in senses]
    return features_compare(df, diagnoses['plus'], features)


def all_compare(df, diagnoses, sense, direction):
    feature = "%s_%s_illness" % (sense, direction)
    diagnoses = {key: value for key, value in diagnoses.items() if key in df['COVID Status (All)'].values}
    return feature_compare(df, diagnoses, feature)


def conf_mat_stats(Xu, y, feature):
    # precision = ppv
    # recall = sensitivity
    # flip all values and do precision = npv
    # flip all values and do recall = specificity
    scorers = [
        ("PPV", precision_score, 0),
        ("Sensitivity", recall_score, 0),
        ("NPV", precision_score, 1),
        ("Specificity", recall_score, 1),
    ]
    scores = pd.DataFrame(
        index=[0, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    )
    scores.index.name = "VAS Cutoff"
    for value in scores.index:
        y_predict = Xu.loc[y.index, feature] <= value
        scores.loc[value, nicify(feature)] = y_predict.mean().round(2)
        for name, scorer, flip in scorers:
            if flip:
                scores.loc[value, name] = scorer(1 - y, 1 - y_predict).round(2)
            else:
                scores.loc[value, name] = scorer(y, y_predict).round(2)
        num = (y_predict & y).sum() / (y_predict & (1-y)).sum()
        denom = ((1-y_predict) & y).sum() / ((1-y_predict) & (1-y)).sum()
        scores.loc[value, "Odds Ratio"] = (num / denom).round(2)
    return scores


@auto_save_table
def conf_mat_stats2(Xu, y, features):
    scorers = [
        #("PPV", precision_score, 0),
        ("Sensitivity", recall_score, 0),
        #("NPV", precision_score, 1),
        ("Specificity", recall_score, 1),
    ]
    scores = pd.DataFrame(
        #index=nicify(list(features))
    )
    scores.index.name = "Feature"
    for feature, cutoff, direction in features:
        if direction == 1:
            y_predict = Xu.loc[y.index, feature] > cutoff
        elif direction == -1:
            y_predict = Xu.loc[y.index, feature] < cutoff
        else:
            raise Exception("")#y_predict = Xu.loc[y.index, feature] < cutoff
        #scores.loc[value, nicify(feature)] = y_predict.mean().round(2)
        if cutoff == 0.5:
            feature_ = nicify(feature)
        else:
            feature_ = '%s < %s' % (nicify(feature), cutoff)
        for name, scorer, flip in scorers:
            if flip:
                scores.loc[feature_, name] = scorer(1 - y, 1 - y_predict).round(2)
            else:
                scores.loc[feature_, name] = scorer(y, y_predict).round(2)
        num = (y_predict & y).sum() / (y_predict & (1-y)).sum()
        denom = ((1-y_predict)&y).sum() / ((1-y_predict)&(1-y)).sum()
        scores.loc[feature_, "Odds Ratio"] = (num / denom).round(2)
    return scores


# Best model, with error bars and p-values
@auto_save_table
def fit_and_coefs(Xu, y, features, C=10):
    ss = StandardScaler()
    thick_features = ["Smell_during_illness", "Days_since_onset"]
    Xs = Xu.copy()
    Xs.loc[:, thick_features] = ss.fit_transform(Xs[thick_features])
    scale = pd.Series(ss.scale_, index=features)
    scale["const"] = 1
    logit = Logit(y, add_constant(Xs[features]))
    results = logit.fit_regularized(
        method="l1", alpha=1 / C, maxiter=100000, trim_mode="size", acc=1e-20
    )

    return statsmodels_to_df(
        results, plot=False, title="Logistic Regression Model", scale=scale, figsize=(3, 7)
    )


@auto_save_table
def big_table(df, s, clusters, sem=True):
    stc = ["Smell", "Taste", "Chemesthesis"]
    features_of_interest = (
        ["Age", "Gender"]
        + ["%s Change" % x for x in stc]
        + ["%s Recovery" % x for x in stc]
        + ["Anosmia/Hyposmia", "Parosmia", "Phantosmia"]
        + nicify([x for x in list(df) if "basic_taste" in x])
        + ["Onset Day", "Days Since Onset"]
    )
    features_of_interest = list(set(features_of_interest))
    recovery_summary = pd.DataFrame(
        index=features_of_interest,
        columns=pd.MultiIndex.from_product(
            [[x.replace("\n", " ") for x in clusters], ("", "C19+", "p", "C19-")]
        ),
    )
    nice_df = df.copy()
    nice_df.columns = nicify(list(df.columns))
    for feature in recovery_summary.index:
        for cluster in clusters:
            z = {}
            for diagnosis, label in [("lab-covid", "C19+"), ("non-covid", "C19-")]:
                z[label] = nice_df.loc[clusters[cluster] & s[diagnosis], feature]
                if feature == "Gender":
                    z[label] = z[label] == "F"
                if sem:
                    denom = np.sqrt(z[label].notnull().sum())
                else:
                    denom = 1
                summary = "%.2g ± %.2g" % (
                    z[label].mean(),
                    z[label].std() / denom,
                )
                recovery_summary.loc[feature, (cluster.replace("\n", " "), label)] = summary
            _, p = mannwhitneyu(z["C19+"], z["C19-"])
            recovery_summary.loc[feature, (cluster.replace("\n", " "), "p")] = "%.2g" % p
            recovery_summary.loc[feature, (cluster.replace("\n", " "), "")] = "|"
    recovery_summary.index = [
        x.replace("Gender", "Fraction Women") for x in recovery_summary.index
    ]

    return recovery_summary.sort_index(axis=1).style.set_properties(
        **{"text-align": "center"}
    ).set_table_styles([dict(selector="th", props=[("text-align", "center")])])


def predict_others(clf, X, Xn, y, s, classes, class_weights='balanced'):
    sample_weight = get_weights(X, y, "balanced")  # -by-country")
    features = list(
        chain(
            *[
                classes["features"][kind]
                for kind in ["chemosensory", "typical", "history", "demographic"]
            ]
        )
    )
    clf.fit(X[features], y, sample_weight=sample_weight)
    p = clf.predict_proba(Xn.loc[s["lab-covid"], features])[:, 1].mean()
    print(
        "%.3g%% of the COVID Diagnosis code 2/3 subjects are predicted to be COVID+."
        % (p * 100)
    )
    p = clf.predict_proba(Xn.loc[s["non-covid"], features])[:, 1].mean()
    print(
        "%.3g%% of the COVID Diagnosis code 5 subjects are predicted to be COVID+."
        % (p * 100)
    )
    p = clf.predict_proba(Xn.loc[s["unknown-covid"], features])[:, 1].mean()
    print(
        "%.3g%% of the COVID Diagnosis code 4 subjects are predicted to be COVID+."
        % (p * 100)
    )
    p = clf.predict_proba(Xn.loc[s["clinical-covid"], features])[:, 1].mean()
    print(
        "%.3g%% of the COVID Diagnosis code 1 subjects are predicted to be COVID+."
        % (p * 100)
    )

@auto_save_table
def yougov_odds_ratios():
    yougov_country_ors = pd.DataFrame()
    ygs = []
    gccr_list = ['Global', 'brazil', 'canada', 'denmark', 'finland', 'france', 'germany', 'italy', 'mexico',
                 'netherlands', 'norway', 'spain', 'sweden', 'united-kingdom', 'united-states']
    
    def fill(yougov_country_ors, z, country):
        c19p_c19n_or = (z.loc["Smell loss", "C19+"] / z.loc["Smell loss", "C19-"]) / (
            z.loc["No smell loss", "C19+"] / z.loc["No smell loss", "C19-"]
        )
        c19p_others_or = (z.loc["Smell loss", "C19+"] / z.loc["Smell loss", "--"]) / (
            z.loc["No smell loss", "C19+"] / z.loc["No smell loss", "--"]
        )
        c19n_others_or = (z.loc["Smell loss", "C19-"] / z.loc["Smell loss", "--"]) / (
            z.loc["No smell loss", "C19-"] / z.loc["No smell loss", "--"]
        )
        n = z.sum().sum()
        p = z["C19+"].sum() / n
        pm = z["C19-"].sum() / n
    
        yougov_country_ors.loc["N", country] = int(n)
        yougov_country_ors.loc["% C19+", country] = p*100
        yougov_country_ors.loc["% C19-", country] = pm*100
        yougov_country_ors.loc["OR (C19+ vs C19-)", country] = c19p_c19n_or
        yougov_country_ors.loc["OR(C19+ vs Not Tested)", country] = c19p_others_or
        yougov_country_ors.loc["OR(C19- vs Not Tested)", country] = c19n_others_or
        yougov_country_ors.loc["p(Smell Loss | C19+)", country] = z.loc["Smell loss", "C19+"] / z["C19+"].sum()
    
    for file in pathlib.Path("data/yougov").iterdir():
        if not file.suffix == '.csv':
            continue
        yg = pd.read_csv(file, low_memory=False)
        yg = fix_yougov(yg)
        yg = pd.crosstab(index=yg["i5_health_3"], columns=yg["i3_health"])
        yg = yg.rename(
            index={"No": "No smell loss", "Yes": "Smell loss"},
            columns={
                "No, I have not": "--",
                "Yes, and I tested negative": "C19-",
                "Yes, and I tested positive": "C19+",
            },
        ).loc[["No smell loss", "Smell loss"], ["--", "C19-", "C19+"]]
        country = file.name.split(".")[0].split("_")[1]
        if country in gccr_list:
            fill(yougov_country_ors, yg, country)
            ygs.append(yg)
    
    yougov_country_ors = yougov_country_ors.T

    yougov_country_ors.loc["!Global"] = (
        yougov_country_ors.mul(yougov_country_ors["N"], axis=0).sum()
        / yougov_country_ors["N"].sum()
    )
    #return zs
    
    yougov_country_ors = yougov_country_ors.T
    yg[:] = np.dstack([yg.values for yg in ygs]).sum(axis=2)
    fill(yougov_country_ors, yg, '!!Global')
    yougov_country_ors = yougov_country_ors.T
    yougov_country_ors['N'] = yougov_country_ors['N'].astype(int)
    yougov_country_ors.index = yougov_country_ors.index.map(lambda x: x.replace('-', ' ').title())
    
    format_ = {'N': '%d', '% C19+': '%.2g', '% C19-': '%.2g', 'p(Smell Loss | C19+)': '%.2g'}
    for col in yougov_country_ors:
        f = format_.get(col, '%.3g')
        yougov_country_ors[col] = yougov_country_ors[col].apply(lambda x: f%x)
    return yougov_country_ors.sort_index()


@auto_save_table
def symptoms_by_status(df, statuses, dtype, classes, formatting='.%3g', drop_features=None):
    features = []
    for pattern in feature_ontology[dtype]:
        r = re.compile(pattern)
        features += list(filter(r.search, list(df)))
    features = list(set(features))
    df = status_map(df, statuses, "COVID-19 Status")
    mean = df.groupby('COVID-19 Status').mean()[features].T
    mean = mean.applymap(lambda x: formatting % x)
    std = df.groupby('COVID-19 Status').std()[features].T
    std = std.applymap(lambda x: ' +/- %.3g' % x)
    for feature in features:
        try:
            if classes['dtypes'][feature] in ['continuous', 'discrete']:
                mean.loc[feature] += std.loc[feature]
        except:
            pass
    mean.index = mean.index.map(nicify)
    if drop_features:
        mean = mean.drop(drop_features)
    return mean.sort_index()
    