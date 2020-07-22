"""Python library for GCCR002"""

from contextlib import contextmanager
from datetime import datetime
import hashlib
from io import StringIO
from IPython.display import display as _display
from itertools import chain, product, combinations_with_replacement
import joblib
import json
import logging
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import networkx as nx
import numpy as np
import pandas as pd
import pathlib
import pickle
import re
from scipy.special import logit
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon, gaussian_kde, chi2_contingency, entropy, norm
import seaborn as sns
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression, RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import auc, roc_curve, roc_auc_score, plot_roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score, get_scorer, make_scorer, SCORERS
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, LeaveOneOut, cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.api import add_constant
from statsmodels.discrete.discrete_model import Logit
import sys
sys.path.append('/home/rgerkin/dev/pyvenn')  #TODO: Turn pyvenn into a pip-installable package
from tqdm.auto import tqdm, trange
from venn import venn3, venn4, venn5, get_labels
import warnings

sns.set(font_scale=1.1)
sns.set_style('whitegrid')
logger = logging.Logger('GCCR002')
known_md5s = {'GCCR002_complete_database.csv': 'd476f67b081dd9d8d8cf1ee0481ad4e8',
              'GCCR002_DATA_COVID_TimeStamp.xlsx': 'aa016d9208fbb44ffd8ce4a2dfe908a4',
              'GCCR002_DATA_COVID_TimeStamp_plusdataJuly.csv': '56922f025047e379bf5cfc8ff2ceed91'}
DATA = pathlib.Path('data')
YOUGOV_CUTOFF_DATE = '2020-07-03'
# In order to guarantee a match to the published figures, we must remove YouGov reported after the manuscript submission date.
# This corresponds to week 11.  To update this figure with new data (collected by YouGov after manuscript submission),
# change max_week to a higher value (e.g. the present day)"""

# For each type (e.g. categorical), a list of regular expressions for features considered to be that type
dtype_ontology = {'categorical': ['Gender', 'GCCR', 'Referred', 'Test_Name'],
                  'discrete': ['Age', 'Days_since_onset', 'Onset_day', 'Completion_day', 'Recovery'],
                  'binary': ['Changes', 'Symptoms', 'Prior_conditions', 'cigarette(!=_f)', 'cigarette_use', 'Resp'],
                  'continuous': ['(?<!did_)(before_)', 'during_', 'after_', 'change_', 'recovery_', 'frequency', 'cigarette(?!_use)'],
                 }

feature_ontology = {'incidental': ['GCCR', 'Test_Name', 'Completion_', 'Referred'],
                    'chemosensory': ['Changes_in', 'Taste', 'Smell', 'Cheme', '_food', '_smell'],
                    'demographic': ['Gender', 'Age', 'Country'],
                    'history': ['Prior_conditions', 'cigarette'],
                    'typical': ['Symptoms', 'Resp', 'Recovery', 'Blocked', 'Onset_', 'Days_']
                   }

timing_ontology = {'incidental': ['GCCR', 'Test_Name', 'Day', '_day', 'Referred'],
                   'demographic': ['Gender', 'Age', 'Country'],
                   'before': ['Prior_conditions', 'before_illness', 'cigarette'],
                   'during': ['Changes_in', 'change_illness', 'during_illness', 'Resp', 'Symptoms'],
                   'after': ['Recovery', 'after_illness', 'recovery_illness']}

# Color scheme
colors = pd.Series(
    index=pd.MultiIndex.from_tuples([], names=["diagnosis", "sense"]), dtype="object"
)
colors.loc["C19+", "Smell"] = "#6699CD"
colors.loc["C19-", "Smell"] = "#a5bcd4"
colors.loc["C19+", "Taste"] = "#ff9900"
colors.loc["C19-", "Taste"] = "#ffce85"
colors.loc["C19+", "Chemesthesis"] = "#009999"
colors.loc["C19-", "Chemesthesis"] = "#5fc7c7"
colors.loc["C19+", "Blocked_nose"] = "#996600"
colors.loc["C19-", "Blocked_nose"] = "#d1a752"


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def get_hash(x):
    return joblib.hash(x)
        
        
def load_all():
    # All of the content loaded here was produced in pre-analysis.ipynb
    with open(DATA / 'processed' / 'data-types.json') as f:
        dtypes = json.load(f)
    df = pd.read_csv(DATA / 'processed' / 'data-clean.csv', dtype=dtypes, index_col=0)
    Xu = pd.read_csv(DATA / 'processed' / 'X-raw.csv', index_col=0).astype('float')
    Xn = pd.read_csv(DATA / 'processed' / 'X-normalized.csv', index_col=0).astype('float')
    #Xu.index = Xu.index.astype(int)
    #Xn.index = Xu.index.astype(int)
    with open(DATA / 'processed' / 'targets.json') as f:
        targets = json.load(f)
        sets = {name: set(ids) for name, ids in targets.items()}
    with open(DATA / 'processed' / 'classes.json') as f:
        classes = json.load(f)
    return df, Xu, Xn, dtypes, sets, classes


def load_raw():
    #file_name = 'GCCR002_DATA_COVID_TimeStamp.xlsx'
    #file_name = 'GCCR002_DATA_COVID_TimeStamp_plusdataJuly.csv'
    #assert_md5(file_name)  # Check that the MD5 hash of the file is as expected
    #if file_name.endswith('.xlsx'):
    #    df = pd.read_excel(file_name)  # Pandas takes forever to load Excel files
    #elif file_name.endswith('.csv'):
    #    df = pd.read_csv(file_name)
    df_ORIGINAL = pd.read_csv(DATA / 'raw' / 'GCCR002_DATA_COVID_TimeStamp.csv')
    df_JULY = pd.read_csv(DATA / 'raw' / 'GCCR002_julydatabase_timestamp_Countryclean_labelscorrect.csv')
    to_drop = ['UniqueID.1', 'UniqueID_1', 'Unnamed: 0', 'Unnamed: 2', 'Country_clean']
    for df_ in [df_ORIGINAL, df_JULY]:
        df_.drop(to_drop, axis=1, errors='ignore', inplace=True)
        df_['Date_of_onset'] = pd.to_datetime(df_['Date_of_onset'])
        df_['Year_of_birth_Time_Stamp'] = pd.to_datetime(df_['Year_of_birth_Time_Stamp'])
    assert not set(df_ORIGINAL['UniqueID']).intersection(set(df_JULY['UniqueID']))
    df = pd.concat([df_ORIGINAL, df_JULY[df_ORIGINAL.columns]])
    df = df.rename(columns={'Chemethesis_before_illness': 'Chemesthesis_before_illness'})
    assert len(set(df['UniqueID'])) == df.shape[0]
    df = df.set_index('UniqueID')
    df = df.drop('UniqueID.1', errors='ignore')
    report_size(df, 'loading')
    return df


def get_md5(file_name):
    """Get MD5 hash of file"""
    with open(file_name, 'rb') as f:
        # read contents of the file
        data = f.read()    
    # pipe contents of the file through
    md5 = hashlib.md5(data).hexdigest()
    return md5


def assert_md5(file_name):
    md5 = get_md5(file_name)
    assert md5 == known_md5s[file_name], "MD5 hashes do not match; file may have been changed."
    
    
def date_to_integer_day(series):
    series = series.dt.dayofyear
    series = series.fillna(-1).astype(int)
    return series


def display(x):
    if isinstance(x, str):
        print(x)
    else:
        _display(x)

        
def interp_index(array1, array2, threshold):
    i = np.searchsorted(array1, threshold)
    a1 = array1[i-1]
    b1 = array1[i]
    a2 = array2[i-1]
    b2 = array2[i]
    return a2 + (b2-a2)*(threshold-a1)/(b1-a1)


def plot_roc(clf, X, y, cv, cv_kwargs=None, weights=None, concat=True, ax=None, name=None, title=None):
    # Plot ROC curve
    roc_aucs = []
    n = cv.get_n_splits()
    cv_kwargs = {} if cv_kwargs is None else cv_kwargs
    if ax is None:
        plt.figure(figsize=(4,4))
        ax = plt.gca()
    y_score = []
    y_true = []
    all_weights = []
    sample_weight_ = get_weights(X, y, weights)
    for i, (train, test) in enumerate(cv.split(X, **cv_kwargs)):
        #sample_weight = get_weights(X.iloc[train], y.iloc[train], weights)
        sample_weight = sample_weight_.iloc[train]
        clf.fit(X.iloc[train, :], y.iloc[train], sample_weight=sample_weight)
        #sample_weight = get_weights(X.iloc[test], y.iloc[test], weights)
        sample_weight = sample_weight_.iloc[test]
        if hasattr(clf, 'predict_proba'):
            y_score_ = clf.predict_proba(X.iloc[test, :])[:, 1]
        else:
            y_score_ = clf.decision_function(X.iloc[test, :])
        if not concat:
            curve = plot_roc_curve(clf, X.iloc[test, :], y.iloc[test],
                                   alpha=(1/np.sqrt(n)), ax=ax,
                                   sample_weight=sample_weight, name='Split %d' % i)
            roc_aucs.append(curve.roc_auc)
        else:
            auc = roc_auc_score(y.iloc[test], y_score_)
            roc_aucs.append(auc)
        y_score += list(y_score_)
        y_true += list(y.iloc[test])
        all_weights += list(sample_weight)
    score = np.mean(roc_aucs)
    if concat:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=all_weights)
        #score = roc_auc_score(y_true, y_score, sample_weight=all_weights)
        if not name:
            name = clf.__class__.__name__.replace('Classifier','').replace('Ridge', 'Linear')
        sens_half = interp_index(fpr, tpr, 0.5)
        spec_half = 1-interp_index(tpr, fpr, 0.5)
        print("%s: Sens50 = %.3g, Spec50 = %.3g" % (name, sens_half, spec_half))
        label = '%s: %.3g' % (name, score) if name else '%.3g' % score
        ax.plot(fpr, tpr, label=label)
    else:
        ax.set_title('AUC = %.3f +/- %.3f' % (score, np.std(roc_aucs)/np.sqrt(n)))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title:
        ax.set_title(title)
    if n <= 10 or concat:
        ax.legend(fontsize=12, loc=4)
    return score
    

def rank_features(clf, X):
    # Rank the features identified by the classifier from most to least important
    key_features = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    # Show the 20 most important
    key_features.index = nicify(list(key_features.index))
    return key_features.to_frame(name='Importance')


def rank_coefs(clf, X, nicify_=True):
    key_features = pd.Series(clf.coef_.ravel(), index=X.columns)
    if hasattr(clf, 'intercept_') and clf.intercept_:
        key_features['Intercept'] = clf.intercept_[0]
    kf = key_features.to_frame(name='Value')
    kf['Magnitude'] = kf['Value'].abs().round(3)
    kf['Sign'] = ['+' if x>=0 else '-' for x in kf['Value']]
    kf = kf.sort_values('Magnitude', ascending=False)
    kf = kf.drop('Value', axis=1)
    kf = kf[kf['Magnitude']>0]
    if nicify_:
        kf.index = nicify(list(kf.index))
    return kf


def compute_score(clf, X, y, cv):
    # Apply cross-validation using this splitter, and check the following fitness metrics
    results = cross_validate(clf, X, y, scoring=['roc_auc'], cv=cv)
    for key in results:
        print(key, results[key].mean())
        
        
def cardinality_filter(X, n, dtype=None):
    cols = []
    for col in X:
        if dtype is None or X[col].dtype == dtype:
            u = X[col].unique()
            if len(u)>=n:
                cols.append(col)
    return cols


def ontology_to_classes(df, ontology, invert=False, add=None):
    if add is None:
        add = []
    unassigned_cols = list(df.drop('id', errors='ignore'))
    if invert:
        classes = {x:[] for x in ontology}
    else:
        classes = {}
    for key, patterns in ontology.items():
        for pattern in patterns:
            r = re.compile(pattern)
            cols = list(filter(r.search, list(df)))
            for col in cols:
                if col in unassigned_cols:
                    if invert:
                        classes[key].append(col)
                    else:
                        classes[col] = key
                    unassigned_cols.remove(col)
    assert len(unassigned_cols)==0, "%s were unassigned." % unassigned_cols
    for kind in add:
        # The above ontology maps each feature to a single class.
        # Additiomal feature_classes below can reuse these features.
        if kind == 'CDC9':
            classes[kind] = ['Symptoms_%s' % x for x in
                                      ['changes_in_smell', 'changes_in_food_flavor', 'fever', 'muscle_aches',
                                       'runny_nose', 'dry_cough', 'diarrhea', 'fatigue', 'difficulty_breathing_/_shortness_of_breath']]
        if kind == 'CDC7':
            classes[kind] = ['Symptoms_%s' % x for x in
                                      ['fever', 'muscle_aches',
                                       'runny_nose', 'dry_cough', 'diarrhea', 'fatigue', 'difficulty_breathing_/_shortness_of_breath']]
        if kind == 'CDC3':
            classes[kind] = ['Symptoms_%s' % x for x in
                                      ['fever', 'dry_cough', 'difficulty_breathing_/_shortness_of_breath']]
        elif kind == 'chemosensory-binary':
            classes[kind] = [x for x in classes['chemosensory'] if 'illness' not in x]
    
    return classes


def get_rccv_score(clf, X, y, feature_classes, classes, weights='balanced'):
    sample_weight = get_weights(X, y, weights)
    features = list(chain(*[feature_classes[x] for x in classes]))
    clf.fit(X[features], y, sample_weight=sample_weight)
    return clf.best_score_.round(3)


def roc(clf, X, y, feature_classes, classes, cv, weights=None, concat=True, ax=None, with_name=True, title=False):
    features = list(chain(*[feature_classes[x] for x in classes]))
    if with_name:
        name = '%s' % '+'.join(classes)
    score = plot_roc(clf, X[features], y, cv, weights=weights, concat=concat, ax=ax, name=name)
    if ax and title:
        if title is True:
            title = '%s' % '+'.join(classes)
        ax.set_title(title)
    return score

    
def country_weights(X, y):
    test_names = [col for col in X if 'Test_' in col]
    sample_weight = y.copy()
    sample_weight[:] = 1
    for test_name in test_names:
        m = X[test_name].mean()  # Allows this to work even on standardized data
        index = X[X[test_name]>m].index
        if len(index):
            weight = compute_sample_weight('balanced', y.loc[index])
            sample_weight.loc[index] = weight
    return sample_weight


def feature_weights(X, y, feature):
    sample_weight = y.copy()
    sample_weight[:] = 1
    m = X[feature].mean()  # Allows this to work even on standardized data
    index = X[X[feature]>m].index
    if len(index):
        weight = compute_sample_weight('balanced', y.loc[index])
        sample_weight.loc[index] = weight
    return sample_weight


def get_weights(X, y, kind):
    if isinstance(kind, pd.Series):
        sample_weight = kind
    elif kind == 'balanced-by-country':
        sample_weight = country_weights(X, y)
    elif kind == 'balanced':
        sample_weight = compute_sample_weight('balanced', y)
    elif kind:
        sample_weight = compute_sample_weight('balanced', X[kind])
    else:
        sample_weight = compute_sample_weight(None, y)
    sample_weight = pd.Series(sample_weight, index=X.index)
    return sample_weight


def table_summarize(X, y, feature):
    y.name = 'COVID status'
    summary = X.join(y).groupby([feature, 'COVID status']).count().sum(axis=1).to_frame().unstack('COVID status')[0]
    return summary.div(summary.sum()).round(2)


def hist_summarize(X, y, feature):
    plt.hist(X.loc[y==1, feature], color='r', bins=30, alpha=0.3, density=True, label='COVID+');
    plt.hist(X.loc[y==0, feature], color='g', bins=30, alpha=0.3, density=True, label='COVID-');
    plt.legend()
    

def report_size(df, action):
    print("Data after %s has %d subjects and %d features" % (action, *df.shape))
    

def qq_plot(X, y, feature):
    x_minus = X[y==0][feature].quantile(np.linspace(0, 1, 101))
    x_plus = X[y==1][feature].quantile(np.linspace(0, 1, 101))
    ax = sns.lineplot(x_minus, x_plus)
    ax.set_xlabel('%s (COVID -)' % feature.replace('_',' '))
    ax.set_ylabel('%s (COVID +)' % feature.replace('_',' '))
    ax.plot([0, max(x_minus)], [0, max(x_plus)], '--')


def pp_plot(X, y, feature, label=True, stabilized=False, ax=None):
    x_minus = X[y==0][feature]
    x_plus = X[y==1][feature]
    minn = min(x_minus.min(), x_plus.min())
    maxx = max(x_minus.max(), x_plus.max())
    s_minus = pd.Series(index=np.linspace(minn-0.001, maxx+0.001, 200), dtype=float)
    s_plus = pd.Series(index=np.linspace(minn-0.001, maxx+0.001, 200), dtype=float)
    s_minus[:] = s_minus.index.map(lambda x: (x_minus<=x).mean())
    s_plus[:] = s_plus.index.map(lambda x: (x_plus<=x).mean())
    if stabilized:
        s_minus = (2/np.pi)*np.arcsin(np.sqrt(s_minus))
        s_plus = (2/np.pi)*np.arcsin(np.sqrt(s_plus))
    D, p = ks_2samp(x_minus, x_plus)
    #S, p = mannwhitneyu(x_minus, x_plus)
    sign = (s_plus - s_minus).mean() > 0
    #print(sign)
    ax = sns.lineplot(s_minus, s_plus, ax=ax,
                      label='%s (D=%.2f)' % (feature.replace('_', ' ').title(), D if sign>0 else -D))
    ax.set_xlabel('COVID -')
    ax.set_ylabel('COVID +')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend(fontsize=11)
    
    
def nicify(name, line_break=False):
    if isinstance(name, list):
        return list(map(nicify, name))
    s = name.replace('_', ' ').title().replace('Symptoms ', '').split('/')[0].strip().replace('Gccr', 'GCCR v')\
               .replace('Illness Y', 'Illness').replace('Before Illness', 'Before').replace('After Illness', 'After')\
               .replace('Prior Conditions None', 'No Prior Conditions')\
               .replace('Basic Tastes ', '').replace('Recovery Y', 'Recovered').replace('Prior Conditions ', '').split('(')[0]\
               .replace('Changes In Smell I Cannot Smell At All', 'Anosmia/Hyposmia')\
               .replace('Changes In Smell Smells Smell Different Than They Did Before', 'Parosmia')\
               .replace("Changes In Smell I Can Smell Things That Aren'T There", 'Phantosmia')\
               .replace('Changes In Smell Sense Of Smell Fluctuates', 'Smell Fluctuation')\
               .replace('During Illness', 'During').replace(' Illness', '')\
               .replace(' That Required Chemotherapy Or Radiation', '+Chemo/Radiation')\
               .replace('Combustible Cigarette', 'Cigarette')\
               .replace('E-Cigarette 30 Day', 'E-Cigarette')\
               .replace(' That Did Not Require Chemotherapy Or Radiation', '-Chemo/Radiation')\
               .replace('Results','').replace('Final','').replace('!','').replace('Version','').split('[')[0]\
               .replace('Const', 'Intercept')\
               .replace('  ', ' ').strip()
    if line_break:
        x = s.rfind(' ')
        s = s[:x] + '\n' + s[x+1:]
    return s


def nicify_labels(ax, x=True, y=True, line_break=True):
    for xy in ['x', 'y']:
        if locals()[xy]:
            # Fix axis labels
            z = getattr(ax, 'get_%slabel' % xy)()
            new = nicify(z, line_break=line_break)
            getattr(ax, 'set_%slabel' % xy)(new)  
            # Fix tick labels
            z = getattr(ax, 'get_%sticklabels' % xy)()
            new = [nicify(zi.get_text(), line_break=line_break)
                   if not zi.get_text().isnumeric() else zi.get_text()
                   for zi in z]
            getattr(ax, 'set_%sticklabels' % xy)(new)        


def fill_impute(df, feature_dtypes, copy=True):
    if copy:
        df = df.copy()
    # Apply the following missing data handling and recasting rules.
    for col, dtype in feature_dtypes.items():
        if dtype == 'categorical':
            df[col] = df[col].fillna('Missing').astype('object')
        elif dtype == 'discrete':
            df[col] = df[col].fillna(df[col].median()).astype(int)
        elif dtype == 'binary':
            df[col] = df[col].fillna(0.5).astype('float')
        elif dtype == 'continuous':
            df[col] = df[col].fillna(df[col].median()).astype('float')
    return df


def plot_violin(X, y, feature, ax):
    y.name = "COVID status"
    Xy = X.join(y)
    sns.violinplot(x="COVID status", y=feature, data=Xy, ax=ax, alpha=0.2)
    ax.set_xlabel('')
    ax.set_xticklabels(['COVID -', 'COVID +'], fontweight='bold')
    ax.set_ylabel(nicify(feature), fontweight='bold')
    
    
def rescale(X):
    # Create a version of X for which every column has mean 0, variance 1.
    X_st = X.copy()
    std_sclr = StandardScaler()
    X_st[:] = std_sclr.fit_transform(X)
    assert np.allclose(X_st.mean(), 0)
    assert all(np.isclose(X_st.var(ddof=0), 1) + np.isclose(X_st.var(ddof=0), 0))

    # Create a version of X for which every column has min 0, max 1.
    mm_sclr = MinMaxScaler()
    X_nm = X.copy()
    X_nm[:] = mm_sclr.fit_transform(X)
    
    return X_st, std_sclr, X_nm, mm_sclr


def lrcv_check(lrcv, X, y, features):
    sample_weight = get_weights(X, y, 'balanced-by-country')
    lrcv.fit(X[features], y, sample_weight=sample_weight)
    return pd.DataFrame(lrcv.scores_[True].mean(axis=0).round(3),
                        index=pd.Series(lrcv.Cs_, name='C'),
                        columns=pd.Series(lrcv.l1_ratios_, name='L1 Ratio'))


def rccv_check(rccv, X, y, features):
    sample_weight = get_weights(X, y, 'balanced-by-country')
    rccv.fit(X[features], y, sample_weight=sample_weight)
    return rccv.best_score_.round(3), rccv.alpha_


def raw_hist(X, y, feature, cumul=False):
    minn = X[feature].min()
    maxx = X[feature].max()
    diff = maxx - minn
    bins = np.linspace(minn-diff*0.01, maxx+diff*0.01, 30)
    X.loc[y==1, feature].hist(density=True, cumulative=cumul, bins=bins, alpha=0.3, label='+')
    X.loc[y==0, feature].hist(density=True, cumulative=cumul, bins=bins, alpha=0.3, label='-')
    plt.legend()
    plt.title(nicify(feature))
    
    
def contingency(X, features, verbose=True):
    z = pd.crosstab(*[X[f] for f in features])
    z.index.name = nicify(z.index.name)
    z.columns.name = nicify(z.columns.name)
    n = z.sum().sum()
    chi2, p, _, _ = chi2_contingency(z)
    k = min(*z.shape)
    if n and k>1:
        v = np.sqrt(chi2/(n*(k-1)))
    else:
        v = None
    if min(z.shape) >= 2 and z.iloc[0, 1] and z.iloc[1, 1]:
        num = (z.iloc[0, 0] / z.iloc[0, 1])
        denom = (z.iloc[1, 0] / z.iloc[1, 1])
        oddsr =  num / denom 
    else:
        oddsr = None
    if verbose:
        print('p = %.2g' % p)
    return z, p, chi2, v, oddsr


def plot_coefs(clf, X, title=''):
    x = rank_coefs(clf, X)
    #x = x.drop('Intercept', errors='ignore')
    threshold = x.drop('Intercept', errors='ignore')['Magnitude'].max()/10
    x = x[x['Magnitude'] > threshold]
    x = x.sort_values('Magnitude', ascending=True)
    x['Pos'] = x.apply(lambda z: z['Magnitude'] if z['Sign']=='+' else None, axis=1)
    x['Neg'] = x.apply(lambda z: z['Magnitude'] if z['Sign']=='-' else None, axis=1)*-1
    try:
        x['Pos'].plot(kind='barh', color='r', label='+')
    except:
        pass
    try:
        x['Neg'].plot(kind='barh', color='b', label='-')
    except:
        pass
    plt.xlabel('Coefficient Magnitude')
    plt.title(title)
    plt.tight_layout()
    
    
def plot_pb_given_a_(X, b, a, restrict=None, ax=None, title=None, color='k', scale=1000, ticks=None):
    if restrict is not None:
        data = X.loc[restrict, [a, b]]
    else:
        data = X[[a, b]]
    data = data.dropna()
    kde = gaussian_kde(data.T)
    if ticks is None:
        ticks = np.linspace(-100, 100, 9)
    a_ticks = [t for t in ticks if (t>=X[a].min() and t<=X[a].max())]
    b_ticks = [t for t in ticks if (t>=X[b].min() and t<=X[b].max())]
    a_support = a_ticks
    b_support = np.linspace(b_ticks[0], b_ticks[-1], 100)
    aa, bb = np.meshgrid(a_support, b_support)
    pab = kde([aa.ravel(), bb.ravel()]).reshape(len(b_support), len(a_support))
    pab = pd.DataFrame(pab, index=b_support, columns=a_support)
    kde = gaussian_kde(data[a])
    pa = pd.Series(kde(a_support), index=a_support)
    pbga = pab.div(pa)
    if ax is None:
        ax = plt.gca()
    for a_tick in a_ticks:
        l2d = ax.plot(b_support, a_tick + scale*(pbga[a_tick]), label=a_tick, color=color)
        color = l2d[0].get_color()
        ax.plot(b_support, np.ones_like(b_support)*a_tick, '--', color=color)
    ax.set_yticks(a_ticks)
    ax.set_xticks(b_ticks)
    ax.tick_params(reset=True, axis='y', length=5, width=1)
    ax.set_xlim(b_ticks[0], b_ticks[-1])
    ax.set_xlabel(nicify(b))
    ax.set_ylabel(nicify(a))
    if title:
        ax.set_title(title)
    #ax.legend()
    return pab, pa


def plot_difference(pab_0, pa_0, pab_1, pa_1, ax=None, crange=(-1, 1), scale=10):
    pbga_0 = pab_0.div(pa_0)
    pbga_1 = pab_1.div(pa_1)
    assert np.allclose(pbga_0.index, pbga_1.index)
    assert np.allclose(pbga_0.columns, pbga_1.columns)
    log2_odds = np.log2(pbga_1 / pbga_0)
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    norm = Normalize(crange[0], crange[1], True)
    cmap = get_cmap('RdBu_r')
    for a_tick in log2_odds.columns:
        color = cmap(norm(log2_odds[a_tick].values))
        l2d = ax.scatter(log2_odds.index, a_tick + (scale*pa_0[a_tick]*2**log2_odds[a_tick]), label=a_tick, c=color, s=1)
        #color = l2d[0].get_color()
        ax.plot(log2_odds.index, np.ones_like(log2_odds.index)*a_tick, '--', color='k')
    cb = plt.colorbar(l2d)
    cb.outline.set_visible(False)
    cb1 = ColorbarBase(cb.ax, cmap=cmap, norm=norm)
    cticks = np.linspace(*crange, 5)
    cb1.set_ticks(cticks)
    cb1.set_ticklabels(['%.2g' % (2**x) for x in cticks])
    cb1.set_label('Odds Ratio')
    #cb.remove()
    ax.set_title('Ratio')
    

def plot_conditionals(X, y, b, a, restrict=None, crange=(-2, 2), scale=10):
    covid = {0: y[y==0].index,
             1: y[y==1].index}    
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
    if restrict is None:
        restrict = y.index
    restrict_0 = covid[0] & restrict
    restrict_1 = covid[1] & restrict
    pba_0, pa_0 = plot_pb_given_a_(X, b, a, restrict=restrict_0, ax=ax[0], title='COVID-', color='b')
    pba_1, pa_1 = plot_pb_given_a_(X, b, a, restrict=restrict_1, ax=ax[1], title='COVID+', color='r')
    ax[1].set_ylabel('')
    ax[2].set_xlabel(ax[1].get_xlabel())
    plot_difference(pba_0, pa_0, pba_1, pa_1, ax=ax[2], crange=crange, scale=scale)
    plt.tight_layout()
    return pba_0, pba_1


def get_matches(X, match_list):
    return [x for x in X if any([m in x for m in match_list])]


def check_lr(X, y, cv, sample_weight=None):
    from sklearn.linear_model import LogisticRegressionCV
    lrcv = LogisticRegressionCV(penalty='elasticnet',
                                l1_ratios = np.linspace(0, 1, 5),
                                Cs = np.logspace(-3, 3, 7),
                                solver = 'saga',
                                scoring = 'roc_auc',
                                cv = cv,
                                max_iter=10000)
    lrcv.fit(X, y, sample_weight=sample_weight)
    return pd.DataFrame(lrcv.scores_[True].mean(axis=0),
                        index=lrcv.Cs_,
                        columns=lrcv.l1_ratios_)


def venn_covid(X, restrict, features, label, figsize=(5, 5)):
    indices = {}
    for feature in features:
        z = X.loc[restrict, feature]
        indices[feature] = set(z[z==1].index)
    labels = get_labels([indices[feature] for feature in features], fill='percent')
    labels = {k: v.replace('(','').replace(')','') for k, v in labels.items()}
    venn3(labels, names=nicify(features), figsize=figsize, fontsize=9)
    plt.gca().get_legend().remove()
    z = X.loc[restrict, features]
    z = z[z.sum(axis=1)==0].shape[0] / z.shape[0]
    plt.title('%s; None of the three = %.1f%%' % (label, z*100))

    
def kde_plot(df, x, restrict, label, color, ax=None, title=None, **kwargs):
    sns.set_style('whitegrid')
    data = df.loc[restrict, x].dropna()
    x_range = (np.min(df[x]), np.max(df[x]))
    ax = sns.kdeplot(data, clip=x_range, color=color,
                     alpha=0.5, label=label, ax=ax, **kwargs)
    ax.set_xlim(*x_range)
    ax.set_xlabel(nicify(x), fontweight='bold')
    ax.set_ylabel('Probabilty density', fontweight='bold')
    if ax:
        ax.set_title(nicify(x) if title is None else title)
    return ax


def joint_plot(df, x, y, restrict, label, maxx=1e-3, cmap='Reds', cbar=False, ax=None):
    sns.set_style('whitegrid')
    data = df.loc[restrict, [x, y]].dropna()
    x_range = (np.min(df[x]), np.max(df[x]))
    y_range = (np.min(df[y]), np.max(df[y]))
    ax = sns.kdeplot(data[x], data[y], shade=True, clip=[x_range, y_range],
                     vmin=0, vmax=maxx, cmap=cmap, shade_lowest=False, alpha=0.5,
                     ax=ax, n_levels=30, cbar=True,
                     cbar_kws={'format': '%.2g',
                               'label': 'Probability density (x1000)',
                               'shrink': 0.8})
    cax = plt.gcf().axes[-1]
    if cbar:
        cbar_ticks = cax.get_yticks()
        cax.set_yticklabels((cbar_ticks*1000).round(2))
    else:
        cax.remove()
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(nicify(x), fontweight='bold')
    ax.set_ylabel(nicify(y), fontweight='bold')
    ax.set_title(label, fontweight='bold')
    return ax


def feature_hist(df, categories, feature, drop=None, bw=5, cut=0, ax=None, title=None, colors='rbgmck'):
    for i, (label, indices) in enumerate(categories.items()):
        ax = kde_plot(df, feature, indices, label, colors[i], lw=3, bw=bw, cut=cut, ax=ax, title=title)
    ax.legend(fontsize=9);
    
    
def feature_contingency(df, categories, feature, drop=None, normalize=None, verbose=True):
    z = df[[feature]].copy()
    for label, indices in categories.items():
        z.loc[indices, 'Group'] = label
    if drop:
        z = z[~z[feature].isin(drop)]
    c = contingency(z, [feature, 'Group'], verbose=verbose)[0]
    if normalize is not None:
        c = c.div(c.sum(axis=normalize), axis=1-normalize).round(2)
    try:
        c.index = [x.replace('\n', ' ') for x in c.index]
    except:
        pass
    try:
        c.columns = [x.replace('\n', ' ') for x in c.columns]
    except:
        pass
    c = c.rename(index={'M': 'Men', 'F': 'Women'})
        
    return c


def feature_compare(df, categories, feature):
    z = pd.DataFrame(index=list(categories),
                     columns=pd.MultiIndex.from_product([categories, ['Δ', 'σ', 'seΔ', 'D', 'p']]))
    z.index.name = nicify(feature)
    for d1 in categories:
        for d2 in categories:        
            x1 = df.loc[categories[d1], feature]
            x2 = df.loc[categories[d2], feature]
            delta = x1.mean() - x2.mean()
            d = cohen_d(x1, x2)
            p = mannwhitneyu(x1, x2).pvalue
            z.loc[d1, (d2, 'Δ')] = "%.2g" % delta
            z.loc[d1, (d2, 'σ')] = "%.2g" % (0 if not d>0 else delta/d)
            z.loc[d1, (d2, 'seΔ')] = "%.2g" % (delta/np.sqrt(len(x1)+len(x2)))
            z.loc[d1, (d2, 'D')] = "%.2g" % d
            z.loc[d1, (d2, 'p')] = "%.2g" % p
    if len(categories)==2:
        d1 = list(categories)[0]
        d2 = list(categories)[1]
        z = z.loc[d1, d2]
        z.name = nicify(feature)
    return z


def features_compare(df, categories, features):
    assert len(categories) == 2
    zs = [feature_compare(df, categories, feature) for feature in features]
    return pd.concat(zs, axis=1)


def hist_or_contingency(df, categories, feature, drop=None, normalize=None):
    if (df.dtypes[feature] != 'object') and (df[feature].max() > 5 or df[feature].min() < -5):
        f = feature_hist
    else:
        f = feature_contingency
    return f(df, categories, feature, drop=None, normalize=None)


def describe_three_clusters(df, feature, s, drop=None, normalize=None):
    smell_loss = (df['Smell_change_illness']<-80)
    smell_recovery = (df['Smell_recovery_illness']>30)
    r = (df['Recovery_y/n']==2) & df.index.to_series().isin(s['covid'])
    categories = {'Recovered Smell': r & smell_loss & smell_recovery,
                  'Nonrecovered Smell': r & smell_loss & ~smell_recovery,
                  'Intact Smell': r & ~smell_loss}
    return hist_or_contingency(df, categories, feature, drop=drop, normalize=normalize)


def get_set(df, query):
    return set(df.query(query, engine='python').index)


def diagnosis_joint_plots(df, feature1, feature2, r, s, maxx=3e-4):
    if r is None:
        r = set(df.index)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 11.5))
    for i, (diagnosis, label, cmap) in enumerate([('lab-covid', 'C19+', 'Reds'),
                                                  ('non-covid', 'C19-', 'Reds')]):
        joint_plot(df, feature1, feature2,
                   r & s[diagnosis], label, cmap=cmap, maxx=maxx, ax=ax[i], cbar=(i==0))
    return ax


def statsmodels_to_df(results, plot=False, title='', figsize=(10, 5), scale=None):
    summ = results.summary()
    df = pd.read_csv(StringIO(summ.tables[1].as_csv()), index_col=0)
    df.columns = df.columns.str.strip()
    df['abs_coef'] = df['coef'].abs()
    df.index = df.index.str.strip()
    df = df.sort_values('abs_coef', ascending=False)
    df = df.round(2)
    df['P>|z|'] = results.pvalues#.apply(lambda x: '%.1g'%x)
    df = df[df['abs_coef']>0]
    if scale is not None and scale is not False:
        df['coef'] /= scale
        try:
            df['std err'] /= scale
            df['[0.025'] /= scale
            df['0.975]'] /= scale
        except:
            pass
    df.index = nicify(list(df.index))
    if plot:
        plt.figure(figsize=figsize)
        dfp = df.drop('Intercept')
        ax = dfp.sort_values('abs_coef', ascending=True).plot.barh(y='coef', xerr='std err', legend=None, capsize=4, ax=plt.gca())
        labels = ax.get_yticklabels()
        ax.set_yticklabels('%s\n(p=%.1g)' % (label.get_text(), df['P>|z|'].iloc[-i-2])
                           for i, label in enumerate(labels))
        ax.set_xlabel('Regression Coefficient', fontweight='bold')
        ax.set_title(title, fontweight='bold')
    df = df.drop('abs_coef', axis=1)
    for col in df:
        def fill(x):
            try:
                return '%.2g' % float(x)
            except:
                return None
        df[col] = df[col].apply(fill)
    return df


def pooled_sd(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    s1 = np.std(x1)
    s2 = np.std(x2)
    num = (n1-1)*(s1**2) + (n2-1)*(s2**2)
    denom = n1 + n2 - 2
    return np.sqrt(num/denom)


def cohen_d(x1, x2):
    return (np.mean(x1) - np.mean(x2)) / pooled_sd(x1, x2)


def sequential_features(clf, X, y, features, cv, Cs=np.logspace(-1, 3, 5)):
    """Return feature that maximizes cross-validated ROC AUC,
    then feature that maximizes it given inclusion of the first feature, and so ob"""
    roc_aucs = pd.DataFrame(columns=pd.MultiIndex.from_product([Cs, ['Rank', 'AUC']]), dtype='float')
    bar0 = tqdm(Cs)
    bar1 = trange(len(features))
    bar2 = trange(len(features))
    for C in bar0:
        clf.C = C
        features_used = []
        features_remaining = features.copy()
        bar1.reset()
        for i in range(len(features)):
            bar1.update(1)
            best_auc = 0
            best_feature = None
            bar2.reset()
            z = pd.Series(index=features_remaining)
            for j in range(len(features)):
                bar2.update(1)
                feature = features[j]
                if feature in features_remaining:
                    auc = cross_val_score(clf, X[features_used + [feature]], y, cv=cv,
                                          scoring='roc_auc', n_jobs=cv.n_splits).mean()
                    #auc += 0.003*('savory' in feature.lower())
                    z[feature] =  auc
                    if auc > best_auc:
                        best_feature = feature
                        best_auc = auc
            features_used.append(best_feature)
            features_remaining.remove(best_feature)
            #print(z.sort_values(ascending=False))
            roc_aucs.loc[best_feature, (C, 'Rank')] = i+1
            roc_aucs.loc[best_feature, (C, 'AUC')] = best_auc
    return roc_aucs


def status_map(df, mapping, name):
    status = {}
    for k, v in mapping.items():
        status.update({key: k for key in v})
    df[name] = df.index.map(status.get)
    return df


def get_tuple_feature_aucs(clf, X, y, n, sample_weight=None, only_binary=False, nicify_=True, add_to=None):
    if only_binary:
        symptoms = [x for x in X if 'Symptoms_' in x]
    else:
        symptoms = list(X)
    if n > 1:
        tuples = combinations_with_replacement(symptoms, n)
    else:
        tuples = symptoms
    s = pd.Series(index=tuples, dtype='float')
    for tup in tqdm(s.index):
        if n > 1:
            tup_ = list(set(tup)) # Get rid of duplicates
        else:
            tup_ = [tup]
        if add_to:
            tup_+= list(set(add_to))
        clf.fit(X[tup_], y, sample_weight=sample_weight)
        s.loc[tup] = clf.scores_[True].mean()
    if n>1:
        s.index = pd.MultiIndex.from_tuples(s.index)
        if nicify_:
            s.index = s.index.map(lambda x: ' + '.join([nicify(xi) for xi in x]))
    else:
        if nicify_:
            s.index = nicify(list(s.index))
    s = s.sort_values(ascending=False).round(3)
    s.index.name = 'Symptom set'
    df = s.to_frame('ROC AUC')
    return df


def yg_week(df_gccr, offset=0, how='Onset_day'):
    days = (datetime.strptime('2020/04/01', '%Y/%m/%d') - datetime.strptime('2020/01/01', '%Y/%m/%d')).days
    days += offset
    return 1 + ((df_gccr[how].astype(int) - days)/7).astype(int).clip(0, 9999)


def download_yougov():
    url = 'https://raw.githubusercontent.com/YouGov-Data/covid-19-tracker/master'
    yg_countries = pd.read_csv('%s/countries.csv' % url, header=None)[0]
    path = pathlib.Path('data/yougov')
    path.mkdir(parents=True, exist_ok=True)
    for country in tqdm(yg_countries):
        yg = pd.read_csv('%s/data/%s.csv' % (url, country.replace(' ', '-').replace('emerites', 'emirates')),
                         encoding='latin1', dtype='object')
        yg.to_csv(path / ('yougov_%s.csv' % country))
    return yg_countries


def fix_yougov(yg):
    yg = yg[yg['qweek'].str.contains('week')].copy()#.dropna(subset=['qweek'])
    yg['week'] = yg['qweek'].apply(lambda x: x.split(' ')[1]).astype(int)
    try:
        yg['endtime'] = pd.to_datetime(yg['endtime'], format='%d/%m/%Y %H:%M')
    except:
        yg['endtime'] = pd.to_datetime(yg['endtime'], format='%Y-%m-%d %H:%M:%S')
    yg = yg[yg['endtime']<=YOUGOV_CUTOFF_DATE]
    return yg
    
def get_yougov(df_gccr, countries):
    df = pd.DataFrame(index=countries)
    for country in countries:
        yg = pd.read_csv('data/yougov/yougov_%s.csv' % country, dtype='object')
        yg = fix_yougov(yg)
        country = 'usa' if country == 'united-states' else country
        country = 'uk' if country == 'united-kingdom' else country
        weights = df_gccr[df_gccr['Country_of_Residence']==country]['yg_week'].value_counts()
        for week in (set(weights.index) | set(yg['week'])):
            if week not in (set(weights.index) & set(yg['week'])):
                weights.loc[week] = 0
        weight = weights[yg['week']]
        z = pd.get_dummies(yg[['i3_health', 'i4_health']])
        p = z[[x for x in z.columns if 'positive' in x]]
        n = z[[x for x in z.columns if 'negative' in x]]
        total = p.sum().sum() + n.sum().sum()
        p = p.mul(weight.values, axis=0).sum().sum()
        n = n.mul(weight.values, axis=0).sum().sum()
        if p+n:
            df.loc[country, 'YG_fp'] = p/(p+n)
        else:
            df.loc[country, 'YG_fp'] = None
        df.loc[country, 'YG_N'] = total
    df = df.drop(['united-states', 'united-kingdom'])
    df.index.name = 'Country'
    df = df.sort_values('YG_fp')
    return df


def compare_yougov(df_gccr, df_yg, s):
    df_gccr['status'] = -1
    #df_gccr.loc[s['lab-covid'] | s['clinical-covid'], 'status'] = 1
    df_gccr.loc[s['lab-covid'], 'status'] = 1
    df_gccr.loc[s['non-covid'], 'status'] = 0
    df_gccr = df_gccr[df_gccr['status'] >= 0]
    #df_gccr['status'] = df_gccr['status'].astype(int)
    df_gccr = df_gccr.groupby('Country_of_Residence').agg({'status': ['mean', 'count']})['status']
    df_gccr.columns = ['GCCR_fp', 'GCCR_N']
    df_gccr = df_gccr[df_gccr['GCCR_N']>=10]
    df = df_gccr.join(df_yg, how='outer')
    return df


def plot_fp(fp, drop=None, plot=True, verbose=False, ax=None, break_axis=False):
    if ax is None and plot:
        ax = plt.gca()
    if drop:
        fp = fp.copy()
        fp = fp.drop(drop)
    for kind in ['GCCR', 'YG']:
        p = fp['%s_fp' % kind]
        n = fp['%s_N' % kind]
        fp['%s_se' % kind] = np.sqrt(p*(1-p)/n)
        fp['%s_logodds_fp' % kind] = np.log(p/(1-p))
    if plot:
        ax.errorbar(fp['GCCR_fp'], fp['YG_fp'], xerr=fp['GCCR_se'], yerr=fp['YG_se'], marker='o', ls='none', alpha=0.5);
        ax.set_xlabel('GCCR Fraction of COVID Tests Positive')
        ax.set_ylabel('YouGov Fraction of\nCOVID Tests Positive');
    fp_ = fp.dropna()
    #lr = LinearRegression()
    #lr.fit(fp_[['GCCR_logodds_fp']], fp_[['YG_logodds_fp']], sample_weight=1/fp_['GCCR_se']**2)
    #x = np.linspace(-10, 10, 1000)
    #y = lr.predict(x.reshape(-1, 1))
    #from scipy.special import expit
    #plt.plot(expit(x), expit(y), '--')
    from scipy.stats import spearmanr, pearsonr
    def pearson_spearman(x, y):
        return pearsonr(x, y)[0], spearmanr(x, y)[0]
    if verbose:
        print("Log-Odds R = %.3g; Rho=%.3g" % pearson_spearman(fp_['GCCR_logodds_fp'], fp_['YG_logodds_fp']))
        print("Raw R = %.3g; Rho=%.3g" % pearson_spearman(fp_['GCCR_fp'], fp_['YG_fp']))
    return pearson_spearman(fp_['GCCR_fp'], fp_['YG_fp'])[0]
    
    
def cluster_summary(df, clusters, s, feature):
    z = pd.DataFrame(index=list(clusters), columns=['female', 'male'])
    for cluster in z.index:
        for gender in z.columns:
            restrict = clusters[cluster] & s[gender]
            mean = df.loc[restrict, feature].mean()
            std = df.loc[restrict, feature].std()
            z.loc[cluster, gender] = '%.3g +/- %.2g' % (mean, std)
    z.index = [x.replace('\n', ' ') for x in z.index]
    z.index.name = nicify(feature)
    z.columns = ['Women', 'Men']
    return z


def exclusive_best_tuples(tuple_feature_aucs):
    z = tuple_feature_aucs.copy()
    z.index = pd.MultiIndex.from_tuples(
                list(tuple_feature_aucs.index.map(lambda x: x.split(' + '))))
    i = 0
    while True:
        index = z.index[i]
        z = z.drop([x for x in z.index[i+1:] if x[0] in index or x[1] in index])
        i += 1
        if i >= z.shape[0]:
            break
    return z


def compare_lr_model_roc_aucs(X, y, cv, feature_sets, Cs):
    z = pd.DataFrame(index=list(Cs), columns=feature_sets.keys())
    z.index.name = 'C'
    for C in Cs:
        # Use the same model again
        lr = LogisticRegression(penalty='elasticnet', solver='saga', C=C,
                                l1_ratio=1, max_iter=10000, random_state=0)
        for label, features in feature_sets.items():
            z.loc[C, label] = cross_val_score(lr, X[features], y, scoring='roc_auc', cv=cv,
                                              n_jobs=cv.n_splits).mean().round(3)
    return z


def single_and_cumulative_plot(single_aucs, single_xrange, cumul_aucs, cumul_xrange, n_features, classes, C=10, figsize=(14, 6)):
    # Figure layout
    #sns.set_style('whitegrid')
    fig = plt.figure(figsize=figsize)
    width_ratios = [single_xrange[1]-single_xrange[0], cumul_xrange[1]-cumul_xrange[0]]
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=width_ratios)

    # First panel
    ax0 = fig.add_subplot(spec[0])
    feature_names = list(single_aucs.index)
    ax0.plot(single_aucs, feature_names, 'ko', markersize=10)
    ax0.hlines(y=range(n_features), xmin=single_xrange[0], xmax=single_aucs,
               color='gray', alpha=0.2, linewidth=5)
    ax0.set_ylim(n_features-0.5, -0.5)
    ax0.set_xlim(*single_xrange)
    ax0.set_xlabel('ROC Area Under Curve')
    #ax0.set_title('Top single-feature models')
    ax0.set_yticklabels(feature_names)
    def fix_ticklabels(axes):
        for ticklabel in axes.get_yticklabels():
            text = ticklabel.get_text()
            ticklabel.set_weight("normal")
            if text in classes['features']['chemosensory']:
                ticklabel.set_weight("bold")
            if classes['dtypes'].get(text, '') not in ['binary', 'categorical']:
                ticklabel.set_style("oblique")
    fix_ticklabels(ax0)
    ax0.set_yticklabels(nicify(feature_names))
    
    # Second panel
    if cumul_xrange[0] == cumul_xrange[1]:
        return 0
    ax1 = fig.add_subplot(spec[1])
    z = cumul_aucs[C].sort_values('Rank')
    z['Rank'] = np.arange(1, z.shape[0]+1)
    z.plot(x='AUC', y='Rank', color='k', marker='o', ax=ax1)
    ax1.set_ylim(n_features+0.5, 0.5)
    ax1.set_ylabel('Cumulative # Features')
    ax1.set_xlabel('ROC AUC')
    ax1.set_xlim(*cumul_xrange)
    ax1.set_yticks(range(1, n_features+1))
    ax1.legend().remove()
    #ax1.set_title('Top cumulative features model')
    twinax = ax1.twinx()
    twinax.set_ylim(n_features+0.5, 0.5)
    twinax.set_yticks(range(1, n_features+1))
    feature_names = list(cumul_aucs[C].sort_values('Rank').index)
    twinax.set_yticklabels(feature_names)
    fix_ticklabels(twinax)
    feature_names = nicify(feature_names)
    twinax.set_yticklabels(['+%s' % f if i else f for i, f in enumerate(feature_names)]);
    plt.tight_layout()
    
    
def single_plot(single_aucs, single_xrange, n_features, classes, figsize=(10, 6), ax=None, delta=False):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    feature_names = list(single_aucs.index)
    ax.plot(single_aucs, feature_names, 'ko', markersize=10)
    ax.hlines(y=range(n_features), xmin=single_xrange[0], xmax=single_aucs,
               color='gray', alpha=0.2, linewidth=5)
    ax.set_ylim(n_features-0.5, -0.5)
    ax.set_xlim(*single_xrange)
    if delta:
        ax.set_xlabel('Δ ROC AUC')
    else:
        ax.set_xlabel('ROC AUC')
    ax.set_yticklabels(feature_names)
    
    def fix_ticklabels(axes):
        for ticklabel in axes.get_yticklabels():
            text = ticklabel.get_text()
            ticklabel.set_weight("normal")
            if text in classes['features']['chemosensory']:
                ticklabel.set_weight("bold")
            if classes['dtypes'].get(text, '') not in ['binary', 'categorical']:
                ticklabel.set_style("oblique")
    fix_ticklabels(ax)
    ax.set_yticklabels(nicify(feature_names))
    return ax
    
    
def plot_cumul_roc_aucs(roc_aucs):
    for C in roc_aucs.columns.levels[0]:
        roc_aucs[C].sort_values('Rank')['AUC'].plot(label='C=%.3g' % C)
    plt.legend()
    ns = range(0, roc_aucs.shape[0], 5)
    plt.xticks(ns, ['%d' % (1+x) for x in ns])
    plt.xlabel('# Features')
    
    
def fix_quartile_lines(ax):
    for i, l in enumerate(ax.lines):
        if i % 3 != 1:
            l.set_linestyle('--')
        else:
            l.set_linestyle('-')
        l.set_linewidth(2)
        if int(i/3) % 2 == 0:
            l.set_color('white')
        else:
            l.set_color('black')
            

def mutual_info(contingency):
    pxy = (contingency+1e-25) / contingency.sum().sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    I = 0
    for i in range(len(px)):
        for j in range(len(py)):
            I += pxy.iloc[i, j] * np.log2(pxy.iloc[i, j] / (px.iloc[i]*py.iloc[j]))
    return I
    
    
def check_splits(clf, X, y, features, s, set_keys):
    for train_keys in combinations_with_replacement(set_keys, len(set_keys)-1):
        train_keys = set(train_keys)
        train = set.union(*[s[x] for x in train_keys])
        for test_keys in combinations_with_replacement(set_keys, len(set_keys)-1):
            test_keys = set(test_keys)
            if not train_keys.intersection(test_keys):
                test = set.union(*[s[x] for x in test_keys])
                clf.fit(X.loc[train, features], y[train])
                auc = roc_auc_score(y[test], clf.predict_proba(X.loc[test, features])[:, 1])
                print(
                    "Trained on: %s; Tested on: %s; AUC=%.3g"
                    % (train_keys, test_keys, auc)
                )


def odds_ratio(df, sets, feature):
    df_c = feature_contingency(df, sets, feature, verbose=False)
    return (df_c.loc[1, "C19+"] / df_c.loc[1, "C19-"]) / (
        df_c.loc[0, "C19+"] / df_c.loc[0, "C19-"]
    )


def compute_gccr_yougov_corr(df, s, yg_countries):
    country_of_residence = pd.read_csv("data/processed/country-of-residence.csv", index_col=0)
    df_ = df.join(country_of_residence, how="inner")
    rs = []
    result = pd.Series(index=np.arange(7 * -6, 7 * 7, 7), dtype='float')
    
    for offset in tqdm(result.index):
        df_["yg_week"] = yg_week(df_, offset=offset, how="Completion_day")
        # df_["yg_week"] = yg_week(df_, offset=offset, how="Onset_day")
        yougov = get_yougov(df_, yg_countries)
        gccr_yougov = compare_yougov(df_, yougov, s)
        if offset == 0:
            # Print sample size at offset=0
            print(gccr_yougov[["GCCR_N", "YG_N"]].dropna().sum().astype(int))
        result[offset] = plot_fp(gccr_yougov, plot=False)
    return result


def get_auc_p(auc, n1, n2):
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    mu_u = n1 * n2 / 2
    u = auc * n1 * n2
    z = (u - mu_u) / sigma_u
    return 1 - norm.cdf(z)


def tranche_compare(clf, X, y, s):
    for gccr in [1, 2, 3]:
        s["gccr%d" % gccr] = get_set(df, "GCCR==%d" % gccr) & (
            s["lab-covid"] | s["non-covid"]
        )
    check_splits(
        clf,
        X,
        y,
        ["Smell_during_illness", "Days_since_onset"],
        s,
        ["gccr1", "gccr2", "gccr3"],
    )
