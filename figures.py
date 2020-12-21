from gccr002 import *
SAVE = True
CONTEXT = None


def save_fig(name):
    dpi = 300
    formats = ['png', 'pdf', 'tif', 'eps']
    if True:
        path = pathlib.Path('figures')
        if isinstance(CONTEXT, int):
            path = path / ('Figure %d' % CONTEXT)
        elif isinstance(CONTEXT, str):
            if CONTEXT.isnumeric():
                path = path / ('Figure %s' % CONTEXT)
            elif CONTEXT[0]=='S' and CONTEXT[1:].isnumeric():
                path = path / ('Figure %s' % CONTEXT)
            else:
                path = path / CONTEXT
        path.mkdir(exist_ok=True, parents=True)
        with all_logging_disabled():
            for format_ in formats:
                if format_ == 'tif':
                    kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}}
                else:
                    kwargs = {}
                # Some versions require string conversion of path
                path_ = str(path / ('%s.%s' % (name, format_)))
                plt.savefig(path_, format=format_, bbox_inches="tight", **kwargs)

    
def auto_save_fig(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        if SAVE:
            jsonable = [x for x in args if isinstance(x, (int, str))]
            jsonable += [x for x in kwargs.values() if isinstance(x, (int, str))]
            h = hashlib.sha1(json.dumps(jsonable).encode("utf-8")).hexdigest()[:10]
            save_fig('%s_%s' % (func.__name__, h))
    return wrapper


@auto_save_fig
def lab_vs_clinical(df, s, senses):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    ax = axes.flat
    for i, sense in enumerate(senses):
        features = [
            "%s_%s" % (sense, when) for when in ["before_illness", "during_illness"]
        ]
        only = s["lab-covid"] | s["clinical-covid"]
        z = df.loc[only].melt(id_vars=["id", "COVID Status (All)"], value_vars=features)
        sns.violinplot(
            data=z,
            x="variable",
            y="value",
            hue="COVID Status (All)",
            inner="quartile",
            ax=ax[i],
            orient="v",
            split=True,
            cut=0,
            palette={
                "C19+": colors.loc["C19+", sense],
                "C19+ (Clinical)": colors.loc["C19-", sense],
            },
        )
        fix_quartile_lines(ax[i])
        nicify_labels(ax[i])
        ax[i].set_xlabel("")
        ax[i].set_ylabel("Rating" if i == 0 else "", fontweight="bold")
        l = ax[i].legend(loc=(0.3, 1.02), fontsize=10)
        l.get_texts()[0].set_text("C19+ (Lab)")
        ax[i].set_yticks(np.linspace(0, 100, 6))
    ax[0].set_yticklabels(np.linspace(0, 100, 6).astype(int))
    ax[0].set_ylim(0, 100)
    plt.tight_layout()


@auto_save_fig
def pc_senses(df, s, senses):
    features = ["%s_change_illness" % sense for sense in senses]
    z = df.loc[s["lab-covid"] | s["clinical-covid"], features]
    pca = PCA(2)
    pca.fit(z)
    sns.set_style("ticks")
    plt.figure(figsize=(4, 4))
    circle = plt.Circle((0, 0), 1, fill=True, color="lightgray", alpha=0.5)
    for i, sense in enumerate(senses):
        x, y = pca.components_[:, i]
        r = np.sqrt(x ** 2 + y ** 2)
        color = colors["C19+", sense]
        plt.arrow(
            0,
            0,
            x / r,
            y / r,
            color=color,
            width=0.03,
            length_includes_head=True,
            alpha=0.5,
        )
        plt.plot([0, 0], [0, 0], c=color, label=nicify(features[i]))
    plt.gca().add_artist(circle)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.legend(fontsize=8, loc=(0, 1.05))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")


@auto_save_fig
def senses_summary(df, s, senses, left=["before", "during"], right=['change']):
    fig, ax = plt.subplots(4, 2, figsize=(6, 10))
    for i, sense in enumerate(senses):
        features = [
            "%s_%s_illness" % (sense, when) for when in left
        ]
        # for j, status in enumerate(['lab-covid', 'non-covid']):
        only = df.index.intersection(s["lab-covid"] | s["non-covid"])
        z = df.loc[only].melt(id_vars=["COVID Status"], value_vars=features)
        sns.pointplot(
            data=z,
            y="value",
            x="variable",
            hue="COVID Status",
            ax=ax[i, 0],
            scale=1.25,
            markersize=25,
            palette={"C19+": colors.loc["C19+", sense], "C19-": colors.loc["C19-", sense]},
        )
        c19plus = df.loc[only & s["lab-covid"], features[1]].dropna()
        c19minus = df.loc[only & s["non-covid"], features[1]].dropna()
        _, p = mannwhitneyu(c19plus, c19minus)
        print('%s (%s): C19+ = %.3g +/- %.3g; C19- = %.3g +/- %.3g; p=%.3g' %
              (nicify(sense), left[1], c19plus.mean(), c19plus.std(), c19minus.mean(), c19minus.std(), p))
        zeros = int(-np.log10(p))
        #stars = "*" * (np.clip(zeros, 2, 5) - 2)
        #ax[i, 0].text(
        #    1.1, (c19plus.mean() + c19minus.mean()) / 2, stars, va="center", fontsize=18
        #)
        # Draw a seaborn pointplot onto each Axes
        ax[i, 0].set_xlabel(nicify(sense))
        ax[i, 0].set_xticklabels([nicify(x).replace(' illness', '').replace(' ', '\n') for x in left])
        ax[i, 0].set_ylabel("Rating" if i == 0 else "")
        ax[i, 0].legend(loc=1, fontsize=9)
        ax[i, 0].set_yticks(np.linspace(0, 100, 6))

        features = [
            "%s_%s_illness" % (sense, when) for when in right
        ]  # , 'recovery_illness']]
        c19plus = df.loc[only & s["lab-covid"], features[-1]].dropna()
        c19minus = df.loc[only & s["non-covid"], features[-1]].dropna()
        _, p = mannwhitneyu(c19plus, c19minus)
        print('%s (%s): C19+ = %.3g +/- %.3g; C19- = %.3g +/- %.3g; p=%.3g' %
              (nicify(sense), right[-1], c19plus.mean(), c19plus.std(), c19minus.mean(), c19minus.std(), p))
        z = df.loc[only].melt(id_vars=["id", "COVID Status"], value_vars=features)
        sns.violinplot(
            data=z,
            x="variable",
            y="value",
            hue="COVID Status",
            cut=0,
            ax=ax[i, 1],
            orient="v",
            split=False,
            palette={"C19+": colors.loc["C19+", sense], "C19-": colors.loc["C19-", sense]},
        )
        nicify_labels(ax[i, 1])
        ax[i, 1].set_xlabel("")
        ax[i, 1].set_ylabel("Δ Rating" if i == 0 else "", fontweight="bold")
        ax[i, 1].legend().remove()
        ax[i, 1].set_yticks(np.linspace(-100, 100, 11))
        ax[i, 0].set_yticklabels(np.linspace(0, 100, 6).astype(int))
        ax[i, 0].set_ylim(0, 100)
        ax[i, 1].set_yticklabels(np.linspace(-100, 100, 11).astype(int))
        ax[i, 1].set_ylim(-100, 100)
        ax[0, 1].set_title('C19+     C19-')
    plt.tight_layout()
    

def alt_senses_summary(df, s, senses):
    fig, ax = plt.subplots(1, 4, figsize=(24, 4), sharey=True)
    for i, sense in enumerate(senses):
        features = [
            "%s_%s" % (sense, when)
            for when in ["before_illness", "during_illness", "after_illness"]
        ]
        only = s["lab-covid"] | s["clinical-covid"] | s["non-covid"]
        z = df.loc[only].melt(id_vars=["id", "COVID Status"], value_vars=features)
        sns.violinplot(
            data=z,
            x="variable",
            y="value",
            hue="COVID Status",
            cut=0,
            ax=ax[i],
            orient="v",
            split=True,
            inner="quartile",
            palette={"C19+": colors.loc["C19+", sense], "C19-": colors.loc["C19-", sense]},
        )
        nicify_labels(ax[i])
        fix_quartile_lines(ax[i])
        ax[i].set_xlabel("")
        ax[i].set_ylabel("Rating" if i == 0 else "", fontweight="bold")
        ax[i].legend(loc=[0.3, 1.02], fontsize=9)
        ax[i].set_yticks(np.linspace(0, 100, 6))
    ax[0].set_yticklabels(np.linspace(0, 100, 6).astype(int))
    ax[0].set_ylim(0, 100)
    

@auto_save_fig
def one_sense_all_diagnoses(df, sense):
    sns.set_style("ticks")
    palette = {
        "C19+": colors.loc["C19+", "Smell"],
        "C19+ (Clinical)": colors.loc["C19-", "Smell"],
        "Unknown": "dimgray",
        "C19-": "lightgray",
    }
    plt.figure(figsize=(8, 4))
    ax = sns.violinplot(
        data=df,
        x="COVID Status (All)",
        y="%s_during_illness" % sense.title(),
        hue="COVID Status (All)",
        orient="v",
        palette=palette,
        cut=0,
        dodge=False,
        inner="quartile",
    )
    fix_quartile_lines(ax)
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_yticklabels(np.linspace(0, 100, 6).astype(int))
    ax.set_ylabel("%s During Illness" % sense.title(), fontweight="bold")
    ax.set_xlabel("")
    ax.legend().remove()  # loc=[1.03, 0.7], fontsize=10);
    ax.text(
        -1.15,
        0,
        "Total\n%s Change" % sense.title(),
        ha="center",
        va="center",
        fontsize=10,
        fontstyle="italic",
    )
    ax.text(
        -1.15,
        100,
        "No\n%s Loss" % sense.title(),
        ha="center",
        va="center",
        fontsize=10,
        fontstyle="italic",
    )
    ax.set_ylim(0, 100)
    

@auto_save_fig
def effect_size_comparison(df, diagnoses, sense, direction):
    feature = '%s_%s_illness' % (sense, direction)
    diagnoses = {key: value for key, value in diagnoses.items() if key in df['COVID Status (All)'].values}
    z = feature_compare(df, diagnoses, feature)
    
    d = z.reorder_levels([1, 0], axis=1)["D"].astype("float")
    for i in range(4):
        for j in range(i, 4):
            d.iloc[i, j] = None

    Δ = z.reorder_levels([1, 0], axis=1)["Δ"].astype("float")
    for i in range(4):
        for j in range(i, 4):
            Δ.iloc[i, j] = None

    sns.set_style("white")
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    sns.heatmap(d.iloc[1:, :3], cmap="Greens", vmin=0, vmax=1, annot=True, ax=ax[0])
    sns.heatmap(Δ.iloc[1:, :3], cmap="Greens", vmin=0, vmax=25, annot=True, ax=ax[1])
    ax[0].set_title("Effect size", pad=15, fontweight="bold")
    ax[1].set_title("Mean difference", pad=15, fontweight="bold")
    for a in ax:
        a.set_ylabel("")
    plt.tight_layout()
    

@auto_save_fig
def numeric_by_categorical(df, s, numeric_feature, categorical_feature, restrict=None):
    only = (s["lab-covid"] | s["non-covid"])
    if restrict:
        only = only & restrict
    ax = sns.violinplot(
        x=categorical_feature,
        y=numeric_feature,
        data=df.loc[only],
        split=True,
        inner="quartile",
        hue="COVID Status",
        palette={"C19+": colors.loc["C19+", "Smell"], "C19-": colors.loc["C19-", "Smell"]},
        cut=0,
        scale_hue=0,
    )
    fix_quartile_lines(ax)
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_yticklabels(np.linspace(0, 100, 6))
    ax.set_ylabel(nicify(numeric_feature), fontweight="bold")
    ax.legend(loc=9)
    ax.set_xlabel("")
    if categorical_feature == 'Gender':
        ax.set_xticklabels(["Women", "Men"])
    ax.set_ylim(0, 100)


@auto_save_fig
def numeric_by_status(df, s, numeric_feature):
    df["junk"] = 1
    only = s["lab-covid"] | s["non-covid"]
    ax = sns.violinplot(
        x="junk",
        y=numeric_feature,
        data=df.loc[only],
        split=True,
        inner="quartile",
        palette={"C19+": colors.loc["C19+", "Smell"], "C19-": colors.loc["C19-", "Smell"]},
        hue="COVID Status",
        cut=0,
    )
    fix_quartile_lines(ax)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Age", fontweight="bold")
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_yticklabels(np.linspace(0, 100, 6).astype(int))
    ax.set_ylim(18, 80)
    ax.legend(loc=1)
    df = df.drop('junk', axis=1)


@auto_save_fig
def numeric_by_status_split(df, s, numeric_feature, split_feature, restrict):
    only = s["lab-covid"] | s["non-covid"]
    if split_feature == "Gender":
        palette = {"F": "darkorange", "M": "#00AA00"}
    else:
        palette = None
    if restrict:
        only = only & restrict
    ax = sns.violinplot(
        x="COVID Status",
        y=numeric_feature,
        data=df.loc[only],
        split=True,
        inner="quartile",
        hue=split_feature,
        palette=palette,
        cut=0,
        scale_hue=False,
    )
    fix_quartile_lines(ax)
    ax.set_xlabel("")
    ax.set_ylabel(numeric_feature, fontweight="bold")
    l = ax.legend(loc=9)
    if split_feature == "Gender":
        l.get_texts()[0].set_text("Women")
        l.get_texts()[1].set_text("Men")
    plt.setp(ax.collections, alpha=0.5)
    

def downsample_auc(clf_cv, X, y, sample_weight=None):
    X_ = X.copy()
    downsample_features = ["Binary"]
    X_["Binary"] = X[
        "Changes_in_smell_i_cannot_smell_at_all_/_smells_smell_less_strong_than_they_did_before"
    ]
    for n in [10, 100, 1000]:
        if n == 1000:
            feature = "Original scale"
        else:
            feature = "%d-point scale" % n
        X_[feature] = (X["Smell_during_illness"] * n).round() / n
        downsample_features.append(feature)
    downsample_aucs = get_tuple_feature_aucs(
        clf_cv, X_[downsample_features], y, 1, sample_weight=sample_weight, nicify_=False
    )
    downsample_features.remove("Original scale")
    ax = downsample_aucs.loc[downsample_features[::-1]].plot.barh()
    ax.legend().remove()
    ax.set_xlabel("ROC AUC")
    ax.set_ylabel("")


@auto_save_fig
def all_rocs(clf, X, y, classes, class_sets, cv, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    aucs = pd.Series(dtype="float")
    for class_set in class_sets:
        aucs["+".join(class_set)] = roc(
            clf,
            X,
            y,
            classes["features"],
            class_set,
            cv,
            weights=None,
            concat=True,
            with_name=True,
            ax=ax
        )
    ax.legend(fontsize=9, loc=4)
    return aucs


@auto_save_fig
def single_auc(single_feature_aucs, classes, figsize=(10, 6)):
    ax = single_plot(single_feature_aucs, [0.5, 0.72], 10, classes, figsize=figsize)
    x1 = single_feature_aucs.loc["Symptoms_sore_throat", "ROC AUC"]
    x2 = single_feature_aucs.loc[
        "Changes_in_smell_i_cannot_smell_at_all_/_smells_smell_less_strong_than_they_did_before",
        "ROC AUC",
    ]
    x3 = single_feature_aucs.loc["Smell_during_illness", "ROC AUC"]
    ax.axvline(x3, color="k", linestyle="--", ymax=21 / 22)
    ax.axvline(x2, color="k", linestyle="--", ymax=12 / 22)
    ax.axvline(x1, color="k", linestyle="--", ymax=5.5 / 22)
    ax.arrow(
        x1,
        7.5,
        (x2 - x1),
        0,
        width=0.1,
        head_width=0.2,
        head_length=0.01,
        length_includes_head=True,
        color="r",
    )
    ax.arrow(
        x2,
        4.5,
        (x3 - x2),
        0,
        width=0.1,
        head_width=0.2,
        head_length=0.01,
        length_includes_head=True,
        color="r",
    )
    ax.text((x1 + x2) / 2, 7.9, "Δ=%.2f" % (x2 - x1), ha="center", fontsize=10)
    ax.text((x2 + x3) / 2, 4.9, "Δ=%.2f" % (x3 - x2), ha="center", fontsize=10)


@auto_save_fig
def single_auc_delta(single_feature_aucs, reference_value, classes, figsize=(10, 6)):
    ax = single_plot(
    single_feature_aucs - reference_value,
    [0, 0.025],
    10,
    classes,
    delta=True,
    figsize=figsize,
)
    ax.set_yticklabels(['+%s' % x.get_text() for x in ax.get_yticklabels()]);


@auto_save_fig
def contour_plots(df, s, feature_x, feature_y, clusters):
    ax = diagnosis_joint_plots(
        df, feature_x, feature_y, s["recovery-data"], s
    )
    ax[0].set_xlim(-100, 25)
    ax[1].set_ylim(-30, 100)
    ax[0].axhline(33.3, linestyle="--", color="k")
    ax[0].axvline(-33.3, linestyle="--", color="k")
    ax[1].axhline(33.3, linestyle="--", color="k")
    ax[1].axvline(-33.3, linestyle="--", color="k")
    coords = {
        "Intact\nSmell": (-6, -18),
        "Recovered\nSmell": (-40, 73),
        "Persistent\nSmell Loss": (-40, -18),
    }
    for kind, members in clusters.items():
        ax[0].text(
            coords[kind][0],
            coords[kind][1],
            "%s\n(%.1f%%)" % (kind, 100*len(members & s["lab-covid"])/len(s["recovery-data"] & s["lab-covid"])),
            ha="right",
            fontweight="normal",
        )
        ax[1].text(
            coords[kind][0],
            coords[kind][1],
            "%s\n(%.1f%%)" % ("\n", 100*len(members & s["non-covid"])/len(s["recovery-data"] & s["non-covid"])),
            ha="right",
            fontweight="normal",
        )
    ax[0].text(
            11,
            90,
            "n=%d" % len(s["recovery-data"] & s["lab-covid"]),
            ha="center",
            fontweight="normal",
            fontsize=10
        )
    ax[1].text(
            11,
            90,
            "n=%d" % len(s["recovery-data"] & s["non-covid"]),
            ha="center",
            fontweight="normal",
            fontsize=10
        )
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    #plt.tight_layout()


@auto_save_fig
def cramer_vs(df, classes, ax=None):
    if ax is None:
        plt.figure(figsize=(2, 12))
        ax = plt.gca()
    discrete_features = [
        feature
        for feature in df
        if classes["dtypes"].get(feature, None) in ["categorical", "binary"]
    ]
    cramer_v = pd.DataFrame(index=discrete_features, dtype=float)
    for feature in cramer_v.index:
        (
            z,
            _,
            _,
            cramer_v.loc[feature, "V"],
            cramer_v.loc[feature, "Odds Ratio"],
        ) = contingency(df, [feature, "COVID Status"], verbose=False)
    # Contingency table lists C19+ first, so ORs are inverted
    cramer_v.loc[:, "Odds Ratio"]  = 1 / cramer_v.loc[:, "Odds Ratio"]
    cramer_v = cramer_v.sort_values("V").dropna()
    cramer_v.index = nicify(list(cramer_v.index))
    cramer_v = cramer_v.drop(["GCCR v"])
    cramer_v["V+"] = cramer_v["V"] * (cramer_v["Odds Ratio"]>1)
    cramer_v["V-"] = cramer_v["V"] * (cramer_v["Odds Ratio"]<1)
    cramer_v["V+"].plot.barh(ax=ax, color='r', label="C19+ OR>1")
    cramer_v["V-"].plot.barh(ax=ax, color='b', label="C19+ OR<1")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cramer V")
    ax.set_xlim(0, 0.25)
    ax.legend(loc=4, fontsize=10)
    
    
def odds_ratios_cutoff(df, diagnoses, feature, binary=None, xlim=None, ax=None):
    if ax is None:
        plt.figure(figsize=(4, 10))
        ax = plt.gca()
    thresholds = [0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 25, 50, 75, 90, 99]
    odds_ratios = pd.Series(index=["Binary"] + thresholds)
    if binary:
        odds_ratios["Binary"] = odds_ratio(
            df,
            diagnoses['clean'],
            binary,
        )
    df_ = df.copy()
    name = nicify(feature)
    for threshold in thresholds:
        df_["VAS Cutoff"] = df[feature] <= threshold
        try:
            odds_ratios[threshold] = odds_ratio(df_, diagnoses['clean'], "VAS Cutoff")
        except:
            odds_ratios[threshold] = None
    odds_ratios.plot.barh(ax=ax)
    ax.set_ylabel("VAS Cutoff (%s)" % name)
    ax.set_xlabel("Odds ratio for C19")
    if xlim:
        ax.set_xlim(*xlim)

        
@auto_save_fig        
def plot_entropy(df, feature, binary=None):
    covid_entropy = entropy(df["COVID Status"].value_counts(normalize=True), base=2)
    if binary:
        binary_entropy = entropy(df[binary].value_counts(normalize=True), base=2)
    else:
        binary_entropy = None
    z = contingency(df, ["COVID Status", "Symptoms_changes_in_smell"], verbose=False)[0]
    binary_mi = mutual_info(z) / covid_entropy

    rounders = [0.1, 1, 2, 5, 10, 20, 50]
    continuous_entropy = {}
    continuous_mi = {}
    for rounder in rounders:
        x = (df[feature] / rounder).round()
        continuous_entropy[rounder] = entropy(x.value_counts(normalize=True), base=2)
        df["rounded"] = x
        z = contingency(df, ["COVID Status", "rounded"], verbose=False)[0]
        continuous_mi[rounder] = (mutual_info(z) / covid_entropy)
    
    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    ax[0].barh(
        ["Binary"] + ["10-point scale"] + ["100-point scale"],
        [binary_entropy, continuous_entropy[10], continuous_entropy[1]],
    )
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Entropy (bits)")
    ax[1].barh(
        ["Binary"] + ["10-point scale"] + ["100-point scale"],
        [binary_mi, continuous_mi[10], continuous_mi[1]],
    )
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Relative Information")
    #ax[1].set_xticks([0, 0.01, 0.02])
    plt.tight_layout()

    
def old_summary(Xu, s):
    # Violin Plots (3 panels, one for before, during, after) like GCCR001 and dots.  Possibly also pp plot.
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    X = Xu.loc[(s["lab-covid"] | s["non-covid"]) & s["recovery-data"]]
    y = X.index.to_series().isin(s["lab-covid"])
    for i, feature in enumerate(
        ["Smell_before_illness", "Smell_during_illness", "Smell_after_illness"]
    ):
        plot_violin(X, y, feature, axes.flat[i])
    pp_plot(X, y, "Smell_before_illness", ax=axes.flat[3])
    pp_plot(X, y, "Smell_during_illness", ax=axes.flat[3])
    pp_plot(X, y, "Smell_after_illness", ax=axes.flat[3])
    axes.flat[3].legend(fontsize=7)
    plt.tight_layout()
    
    
def chem_and_block(df, s):
    r = s["lab-covid"] & s["recovery-data"] & s["resp-recovery"]
    chem_block_categories = {
        "None": s["no_chem_or_block"] & r,
        "B": s["no_chem_but_block"] & r,
        "C": s["chem_no_block"] & r,
        "B&C": s["chem_and_block"] & r,
    }
    feature_hist(df, chem_block_categories, "Smell_recovery_illness", bw=1, title="")
    
    plt.figure()
    df = status_map(df, chem_block_categories, "chem-block")
    sns.set_style("whitegrid")
    only = (s["male"] | s["female"]) & (s["lab-covid"] | s["non-covid"])
    ax = sns.violinplot(
        x="chem-block",
        y="Age",
        data=df.loc[only],
        split=True,
        inner="quartile",
        hue="Gender",
        palette={"F": "darkorange", "M": "green"},
        cut=0,
        scale_hue=0,
    )
    fix_quartile_lines(ax)
    ax.set_ylim(18, 80)
    ax.set_ylabel(nicify(ax.get_ylabel()), fontweight="bold")
    ax.set_xlabel("")
    labels = {"None": "None", "B": "Blockage", "C": "Chemesthesis", "B&C": "Both"}
    ax.set_xticklabels([labels[x.get_text()] for x in ax.get_xticklabels()])
    l = ax.legend(loc=(1.02, 0.81))
    l.get_texts()[0].set_text("Women")
    l.get_texts()[1].set_text("Men")
    

@auto_save_fig
def plot_yougov(rs, ax=None):
    if ax is None:
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
    rs_ = rs.copy()
    rs_.index = -rs.index
    rs_.plot(linestyle='-', marker="o", ax=ax)
    ax.set_xticks(range(-35, 36, 7))
    ax.set_xticklabels([int(x / 7) for x in ax.get_xticks()])
    ax.set_xlim(-43, 43)
    ax.set_xlabel("YouGov Response Week - GCCR Response Week")
    ax.set_ylabel("Correlation in C19+%\nbetween GCCR and YouGov")
    
    
def nmf_cluster(Xn, n_components, classes):
    nmf = NMF(n_components, max_iter=10000)
    X_cs = Xn[classes["features"]["chemosensory"]]
    X_cs = X_cs.drop(
        [x for x in X_cs if "after_ill" in x or "before_ill" in x or "during_ill" in x],
        axis=1,
    )
    nmf.fit(X_cs)
    nmf_weights = pd.DataFrame(nmf.components_.T, index=X_cs.columns).round(1)
    s = sns.clustermap(
        nmf_weights,
        cmap="Greens",
        xticklabels=range(1, nmf.n_components + 1),
        yticklabels=nicify(list(nmf_weights.index)),
        col_cluster=False,
        figsize=(4, 7),
        annot=True,
        annot_kws={"fontsize": 9},
    )
    s.cax.remove()
    plt.xlabel("NMF Component")


def draw_ontology(df, classes):
    pos = {}
    node_colors = {}
    colors = 'rgbym'
    for key, values in classes.items():
        if 'CDC' in key:
            continue
        i = list(classes).index(key)
        pos[key.upper()] = (0.5+i*0.5, 1)
        node_colors[key.upper()] = colors[i]
        for name in values:
            name_ = nicify(name)
            pos[name_] = (0.5+i*0.5, -1*(1+classes[key].index(name)))
            node_colors[name_] = colors[i]
    g = nx.Graph({key.upper(): nicify(value) for key, value in classes.items() if 'CDC' not in key})
    plt.figure(figsize=(20, 20))
    nx.draw(g, pos=pos, with_labels=True, font_size=15, node_shape="s",
            node_color=list(node_colors.values()), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.xlim(0, 3);


@auto_save_fig    
def age_and_gender(df, s, clusters):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    for i in range(2):
        if i == 0:
            only = s["lab-covid"] & (s["male"] | s["female"])
        elif i == 1:
            only = s["non-covid"] & (s["male"] | s["female"])
        df = status_map(df, clusters, "status")
        sns.violinplot(
            x="status",
            y="Age",
            data=df.loc[only].sort_values("Gender"),
            split=True,
            inner="quartile",
            order=list(clusters),
            hue="Gender",
            palette={"F": "darkorange", "M": "green"},
            scale_hue=False,
            cut=0,
            ax=ax[i],
        )
        fix_quartile_lines(ax[i])
        ax[i].set_title("COVID %s" % ("+" if i == 0 else "-"), fontweight="bold")
        if i == 1:
            l = ax[i].legend(loc=[0.3, 0.8])
            l.get_texts()[0].set_text("Women")
            l.get_texts()[1].set_text("Men")
        else:
            ax[i].legend().remove()
        ax[i].set_xlabel("")
        plt.setp(ax[i].collections, alpha=0.5)
    ax[0].set_ylabel("Age", fontweight="bold")
    ax[1].set_ylabel("")
    ax[0].set_ylim(18, 85)
    
    
def predict_young_old(df, clf, X, y, s):
    clf.fit(X, y)
    thick_features = (
        rank_coefs(clf, X, nicify_=False).head(10).index.drop("Intercept", errors="ignore")
    )

    ages = df["Age"].quantile(np.arange(0.05, 1, 0.05)).values
    age_roc_auc = pd.DataFrame(index=ages)
    for age in tqdm(ages):
        s["old"] = get_set(df, "Age>%f" % age)
        s["young"] = get_set(df, "Age<=%f" % age)
        assert not (s["old"] & s["young"])
        X_old = X.loc[s["old"] & (s["lab-covid"] | s["non-covid"])]
        X_young = X.loc[s["young"] & (s["lab-covid"] | s["non-covid"])]
        y_old = X_old.index.to_series().isin(s["lab-covid"])
        y_young = X_young.index.to_series().isin(s["lab-covid"])
        clf.fit(X_old[thick_features], y_old)
        auc = roc_auc_score(y_young, clf.predict_proba(X_young[thick_features])[:, 1])
        age_roc_auc.loc[age, "TestYoung"] = auc
        age_roc_auc.loc[age, "YoungC19+"] = len(X_young.index.intersection(s["lab-covid"]))
        age_roc_auc.loc[age, "YoungC19-"] = len(X_young.index.intersection(s["non-covid"]))
        clf.fit(X_young[thick_features], y_young)
        auc = roc_auc_score(y_old, clf.predict_proba(X_old[thick_features])[:, 1])
        age_roc_auc.loc[age, "TestOld"] = auc
        age_roc_auc.loc[age, "OldC19+"] = len(X_old.index.intersection(s["lab-covid"]))
        age_roc_auc.loc[age, "OldC19-"] = len(X_old.index.intersection(s["non-covid"]))
    age_roc_auc[["TestYoung", "TestOld"]].plot()
    plt.xlabel("Age")
    plt.ylabel("ROC AUC")
    plt.legend()
    
    
def plot_double_features(double_features):
    z = double_feature_aucs.copy()
    z.index = pd.MultiIndex.from_tuples(list(double_feature_aucs.index.map(lambda x: x.split(' + '))))
    z = z['ROC AUC'].unstack()
    z = z.fillna(0) + z.T.fillna(0)
    z.values[np.diag_indices_from(z)] /= 2
    plt.figure(figsize=(20, 15))
    sns.heatmap(z, xticklabels=z.index, yticklabels=z.index, cmap='Reds', 
                vmin=0.5, vmax=0.72, cbar_kws={'label': 'ROC Area Under Curve'});
    

def fig_letters(ax, n, x=-0.15, y=1.02):
    for i in range(n):
        ax[i].text(
            x,
            y,
            "ABCDEFGHIJK"[i],
            transform=ax[i].transAxes,
            fontweight="bold",
            fontsize=18,
        )


@auto_save_fig        
def collider_bias():
    # Make layout
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9.25, 8))

    # Make data for panels A and B
    beta_samples = np.random.beta(1, 3, size=(1000, 2))
    df = pd.DataFrame(beta_samples, columns=['Likelihood of COVID', 'Likelihood of Smell Loss'])

    # Panel A
    df.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[0, 0], alpha=0.5, c='black')
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title('r=0')

    # Panel B
    df_gccr = df[df.sum(axis=1)>0.5]
    df.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[0, 1], alpha=0.1, c='red')
    df_gccr.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[0, 1], alpha=0.5, c='black')
    ax[0, 1].set_title('r=%.2g' % df_gccr.corr().iloc[0, 1])

    # Make data for panels C and D
    from scipy.stats import norm, beta
    cov = [[1, 0.5], [0.5, 1]]
    chol = np.linalg.cholesky(cov)
    corr_normal_samples = (chol @ np.random.randn(2, 1000)).T
    corr_beta_samples = beta.ppf(norm.cdf(corr_normal_samples), 1, 3)
    df2 = pd.DataFrame(corr_beta_samples, columns=['Likelihood of COVID', 'Likelihood of Smell Loss'])

    # Panel C
    df2.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[1, 0], alpha=0.5, c='black')
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_title('r=%.2g'% df2.corr().iloc[0, 1])

    # Panel B
    df2_gccr = df2[df2.sum(axis=1)>0.5]
    df2.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[1, 1], alpha=0.1, c='red')
    df2_gccr.plot.scatter(x='Likelihood of COVID', y='Likelihood of Smell Loss', ax=ax[1, 1], alpha=0.5, c='black')
    ax[1, 1].set_title('r=%.2g' % df2_gccr.corr().iloc[0, 1])

    # Label axes
    for i in range(2):
        ax[i, 0].set_ylabel('Likelihood of Smell Loss')
        ax[1, i].set_xlabel('Likelihood of COVID')

    # Add letters
    fig_letters(ax.flat, 4, x=-0.25)

    # Tidy up
    plt.tight_layout()


@auto_save_fig    
def odds_figure(Xu, y, feature):
    X_ = Xu.loc[y.index, feature]
    target_quantiles = X_.quantile(np.arange(0.005, 1, 0.01))
    cutoff_quantiles = X_.quantile(np.arange(0, 1, 0.01))
    z = cutoff_quantiles.searchsorted(X_)
    X_rounded = target_quantiles.iloc[z].values

    # Doing it the long way not the sns.lineplot way because
    # we need to transform the error bars to odds space
    sns.set_style("ticks")
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 8))
    z = pd.DataFrame(np.vstack((X_rounded, y)).T, columns=["%s_rounded" % feature, "C19"])
    p = z.groupby("%s_rounded" % feature).mean()["C19"]
    n = z.groupby("%s_rounded" % feature).count()["C19"]
    p_low = p - 2 * np.sqrt(p * (1 - p) / n)
    p_high = p + 2 * np.sqrt(p * (1 - p) / n)
    p_high = p_high.clip(0, 1)
    ax[0].fill_between(p.index, p_low.values, p_high.values, color="gray", alpha=0.3)
    ax[0].plot(p.index, p.values, c="black")
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(0, 1)
    ax[0].axhline(0.5, linestyle="--", color="black", alpha=0.5)
    ax[0].set_xlabel(nicify(feature))
    ax[0].set_ylabel("p(C19+)")

    odds = p / (1 - p)
    odds_low = p_low / (1 - p_low)
    odds_high = p_high / (1 - p_high)
    ax[1].fill_between(
        odds.index, odds_low.values, odds_high.values, color="lightblue", alpha=0.3
    )
    ax[1].plot(odds.index, odds.values, c="blue")
    ax[1].set_ylim(0.1, 100)
    ax[1].axhline(1, linestyle="--", color="black", alpha=0.5)
    ax[1].set_yscale("log")
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax[1].set_xlabel(nicify(feature))
    ax[1].set_ylabel("odds(C19+)")
    fig_letters(ax, 2, x=-0.25, y=1.05)


@auto_save_fig 
def odds_cartoon(change_point=3, constant=1.2, slope=-0.33):
    x = np.linspace(0, change_point, 100)
    steady_state = 10 ** (slope * change_point + constant)
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, 10 ** (slope * x + constant), color="blue")
    ax.plot(x, 10 ** (slope * x + constant) / steady_state, color="brown")
    
    x = np.linspace(change_point, 10, 100)
    ax.plot(x, np.ones(100) * steady_state, linestyle="-", color="blue", label='odds (C19+)')
    ax.plot(x, np.ones(100) * 1, linestyle="-", color="brown", label='odds ratio (C19+)')
    ax.axhline(1, linestyle="--", color="black", alpha=0.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel('ODoR-19 Scale')
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.legend()
    #ax[2].set_ylabel("odds(C19+)")
    plt.tight_layout()
    
    
def plot_clipped_auc(clfcv, Xu, y, feature, ns=np.logspace(-1, 2, 16)):
    clipped_auc = pd.Series(index=np.logspace(-1, 2, 16), dtype="float")
    for n in tqdm(clipped_auc.index):
        X_clipped = Xu.loc[y.index, [feature]].clip(0, n)
        clfcv.fit(X_clipped, y)
        clipped_auc[n] = clfcv.scores_[True].mean()
    clipped_auc.plot()
    plt.xlabel("%s\ntruncated below this value" % nicify(feature))
    plt.ylabel("ROC AUC")


@auto_save_fig
def feature_vs_days(df, s, restrict=None, features=["Smell_change_illness", "Smell_recovery_illness"]):
    ax = plt.gca()
    if len(features)>1:
        twinax = ax.twinx()
    for i, feature in enumerate(features):
        sns.lineplot(
            x="Days_since_onset",
            y=feature,
            data=df if restrict is None else df.loc[restrict],
            ax=ax if i==0 else twinax,
            color='blue' if i==0 else 'darkgoldenrod',
        )
    ax.set_xlim(5, 40)
    ax.set_ylim(-100, -45)
    twinax.set_ylim(0, 55)
    #ax.set_yticklabels([x / 2 for x in ax[0].get_yticks()])
    # ax[0, 1].set_ylim(0, 2)
    # ax[0].set_title("C19+")
    # ax[0, 1].set_title("C19-")
    ax.set_xlabel("Days Since Onset")
    # ax[3, 1].set_xlabel("Days Since Onset")
    ax.set_ylabel(nicify(features[0]), color='blue')
    if len(features)>1:
        twinax.set_ylabel(nicify(features[1]), color='darkgoldenrod')
    plt.tight_layout()
    #fig_letters(ax, 4, x=-0.25)
    
    
def which_cluster(df, classes, s, clusters):
    discrete_features = [
        feature
        for feature in df
        if classes["dtypes"].get(feature, None) in ["categorical", "binary"]
    ]
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(5, 10))
    three_clusters_pos = {
        key: clusters[key] & s["lab-covid"] for key in clusters
    }
    two_clusters_pos = {
        key: value
        for key, value in three_clusters_pos.items()
        if key in ["Persistent\nSmell Loss", "Recovered\nSmell"]
    }
    for i, cp in enumerate([three_clusters_pos, two_clusters_pos]):
        df = status_map(df, cp, "Smell Recovery Status")
        cramer_v = pd.Series(index=discrete_features, dtype=float)
        for feature in cramer_v.index:
            z, _, _, cramer_v[feature], _ = contingency(
                df, [feature, "Smell Recovery Status"], verbose=False
            )
        cramer_v.index = nicify(list(cramer_v.index))
        cramer_v.plot.barh(ax=ax[i])
    ax[0].set_title("Which cluster\n(out of 3)?")
    ax[1].set_title("Given smell loss,\nwhich cluster\n(out of 2)?")
    ax[1].set_xlim(0, 1)
    ax[1].set_yticklabels([])
    ax[0].set_xlabel("Cramer V")
    ax[1].set_xlabel("Cramer V")

@auto_save_fig
def c19_by_country(df, s):
    """DataFrame must still have the 'County_of_Residence' and 'COVID_diagnosis' columns,
    i.e. this must be run during pre-analysis before these columns are dropped."""
    sns.set_style('ticks')
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 6))
    z = df.copy()
    z['C19+'] = z['COVID_diagnosis'].isin([2, 3])
    p = z.groupby('Country_of_Residence').mean()['C19+']
    n = z.groupby('Country_of_Residence').count()['C19+']
    topN = n.sort_values().tail(25)
    x = topN.index
    ax[0].barh(x, p[x], xerr=np.sqrt(p[x]*(1-p[x])/n[x]))
    ax[0].set_xlabel('C19+ / (all respondents)')
    ax[0].set_xlim(0, 1)
    ax[0].set_yticklabels(['%s (%d)' % (country.title(), n[country]) for country in x])
    z = z.loc[s['lab-covid'] | s['non-covid']]
    p = z.groupby('Country_of_Residence').mean()['C19+']
    n = z.groupby('Country_of_Residence').count()['C19+']
    ax[1].barh(x, p[x], xerr=np.sqrt(p[x]*(1-p[x])/n[x]))
    ax[1].set_xlabel('C19+ / (C19+ and C19-)')
    ax[1].set_yticklabels(['%s (%d)' % (country.title(), n[country]) for country in x])
    plt.tight_layout()
    
    

    
    
    