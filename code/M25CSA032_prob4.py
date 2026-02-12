import os
import sys
from math import pi

# ---------------------------------------------------------------
# NLU Assignment 1 – Problem 4: Sports vs Politics Classifier
# Zenith | M25CSA032
#
# This is the big one — an end-to-end text classification pipeline
# that loads a sports/politics subset of 20 Newsgroups, tries out
# a bunch of feature+model combos, and spits out nice-looking
# charts + a CSV of all the results.
#
# I went with scikit-learn for the ML side since the assignment
# allows external libraries for this problem.  The main goal is
# comparing at least 3 different ML techniques, so I tried five:
# Naive Bayes, SVM, Logistic Regression, Random Forest, and
# Gradient Boosting.
#
# Results and plots get saved under nlp_final_results/ — the
# LaTeX report references these files directly.
# ---------------------------------------------------------------

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')          # so it works on headless servers
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rcParams

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

except ImportError as e:
    print("Missing library:", e)
    print("Install with:  pip install --user scikit-learn pandas matplotlib seaborn")
    sys.exit(1)


# ---------- colour palette & style setup ----------
# I wanted the charts to look cohesive, so I picked a palette inspired
# by sunset/ocean tones — warm corals for one class, cool teals for
# the other, and a rich gradient for the ranking chart.

CORAL      = '#FF6B6B'
SOFT_RED   = '#EE5A6A'
TEAL       = '#4ECDC4'
DARK_TEAL  = '#2C8C85'
GOLD       = '#FFD93D'
DEEP_NAVY  = '#1A1A2E'
SLATE      = '#34495E'
LIGHT_BG   = '#F8F9FA'

# gradient for bar charts (goes from warm coral to cool teal)
def _bar_gradient(n):
    """Generate n colours that smoothly transition coral → teal."""
    return [
        plt.cm.coolwarm(x) for x in np.linspace(0.15, 0.85, n)
    ]

# set a clean modern look for all plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12
rcParams['axes.facecolor'] = LIGHT_BG
rcParams['figure.facecolor'] = '#FFFFFF'
rcParams['axes.edgecolor'] = '#CCCCCC'
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['grid.color'] = '#CCCCCC'

sns.set_context("talk")

SAVE_DIR = "nlp_final_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------- data loading ----------
def load_data():
    """
    Grab the sports and politics subsets from 20 Newsgroups.
    I'm using 2 sports groups and 2 politics groups, then mapping
    them to a simple binary label.
    """
    print("Loading 20 Newsgroups (Sports vs Politics)...")

    categories = [
        'rec.sport.hockey', 'rec.sport.baseball',
        'talk.politics.guns', 'talk.politics.mideast'
    ]

    dataset = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
    )

    # anything starting with "rec" → Sports, rest → Politics
    labels = [
        'Sports' if dataset.target_names[t].startswith('rec') else 'Politics'
        for t in dataset.target
    ]

    return dataset.data, labels


# ---------- experiment config ----------
def get_configs():
    """
    All the feature/model combos I want to compare.
    Keeping this in one place makes it easy to add or remove
    experiments without touching the rest of the code.
    """
    classifiers = {
        'NB':           MultinomialNB(),
        'LR':           LogisticRegression(max_iter=1000),
        'SVM':          SVC(kernel='linear'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradBoost':    GradientBoostingClassifier(n_estimators=50, random_state=42),
    }

    feature_sets = {
        'BoW':            CountVectorizer(ngram_range=(1, 1)),
        'BoW Bigram':     CountVectorizer(ngram_range=(2, 2)),
        'TF-IDF':         TfidfVectorizer(ngram_range=(1, 1)),
        'TF-IDF Bigram':  TfidfVectorizer(ngram_range=(2, 2)),
        'TF-IDF Trigram': TfidfVectorizer(ngram_range=(3, 3)),
        'N-gram(1,2)':    CountVectorizer(ngram_range=(1, 2)),
    }

    # which (feature, model) pairs to actually run
    experiments = [
        ('BoW', 'NB'),   ('BoW', 'LR'),   ('BoW', 'SVM'),
        ('BoW Bigram', 'RandomForest'),
        ('TF-IDF', 'SVM'), ('TF-IDF', 'NB'), ('TF-IDF', 'LR'),
        ('TF-IDF Bigram', 'SVM'),
        ('TF-IDF Trigram', 'GradBoost'),
        ('N-gram(1,2)', 'SVM'), ('N-gram(1,2)', 'NB'),
    ]

    return classifiers, feature_sets, experiments


# ---------- training & evaluation ----------
def run_experiments(X_train, X_test, y_train, y_test):
    """
    Loop through every feature/model combo, train, predict, and
    collect accuracy + F1.  Also stash the confusion matrices so
    we can plot the best one later.
    """
    clfs, feats, combos = get_configs()
    rows = []
    cm_dict = {}

    print("Running experiments...")

    for feat_name, clf_name in combos:
        print(f"  → {feat_name} + {clf_name}")

        vec = feats[feat_name]
        try:
            X_tr = vec.fit_transform(X_train)
            X_te = vec.transform(X_test)

            model = clfs[clf_name]
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)

            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, pos_label='Politics', average='weighted')

            config_name = f"{feat_name} + {clf_name}"
            rows.append({
                'Configuration': config_name,
                'Feature': feat_name,
                'Classifier': clf_name,
                'Accuracy': acc,
                'F1-Score': f1,
            })
            cm_dict[config_name] = confusion_matrix(y_test, preds, labels=['Sports', 'Politics'])

        except ValueError as e:
            print(f"  ✗ Skipped {feat_name} + {clf_name}: {e}")

    return pd.DataFrame(rows), cm_dict


# ========== VISUALISATIONS ==========
# I tried to make these look polished enough for the report —
# consistent colours, clean labels, and decent resolution (300 dpi).


def plot_top_keywords(X, y, category):
    """
    Horizontal bar chart of the 15 most frequent words in a given
    category (Sports or Politics).  Gives a nice intuition for what
    vocabulary defines each class.
    """
    print(f"  → Top keywords chart for {category}")

    texts = [t for t, lbl in zip(X, y) if lbl == category]
    cv = CountVectorizer(stop_words='english', max_features=15)
    counts_mat = cv.fit_transform(texts)
    word_counts = counts_mat.sum(axis=0).A1
    words = cv.get_feature_names_out()

    df = pd.DataFrame({'word': words, 'count': word_counts})
    df = df.sort_values('count', ascending=True)     # ascending for horizontal bars

    # pick palette based on category
    palette = sns.color_palette('YlOrRd_r', n_colors=len(df)) if category == 'Sports' \
              else sns.color_palette('GnBu_r', n_colors=len(df))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df['word'], df['count'], color=palette)
    ax.set_title(f'Top 15 Words — {category}', fontsize=18, fontweight='bold', color=SLATE)
    ax.set_xlabel('Frequency', fontweight='bold', color=SLATE)
    ax.tick_params(colors=SLATE)

    # add count labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(df['count']) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{int(w)}', va='center', fontsize=10, color=SLATE)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/00_top_words_{category.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_donut_chart(y):
    """Donut chart showing the class balance in the dataset."""
    print("  → Class distribution donut chart")
    labels_unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [CORAL, TEAL]

    wedges, texts, autotexts = ax.pie(
        counts, labels=labels_unique, autopct='%1.1f%%',
        startangle=140, colors=colors,
        pctdistance=0.82, explode=(0.04, 0.04),
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2),
        textprops={'fontsize': 14, 'fontweight': 'bold', 'color': SLATE},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(13)

    ax.set_title('Dataset Class Balance', fontsize=17, fontweight='bold', color=SLATE, pad=20)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/01_class_distribution_donut.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_ranking(df):
    """
    Horizontal bar chart ranking every feature+model combo by accuracy.
    The gradient goes from warm (lower accuracy) to cool (higher),
    so the best models visually "pop" in teal.
    """
    print("  → Accuracy ranking chart")
    df_sorted = df.sort_values('Accuracy', ascending=True)
    n = len(df_sorted)

    # gradient: warm coral → cool teal
    colors = [
        (CORAL if i < n // 3 else GOLD if i < 2 * n // 3 else TEAL)
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(df_sorted['Configuration'], df_sorted['Accuracy'],
                   color=colors, edgecolor='white', linewidth=0.8)

    # zoom into the interesting range
    lo = max(0.60, df_sorted['Accuracy'].min() - 0.03)
    ax.set_xlim(lo, 1.005)
    ax.set_xlabel('Accuracy', fontweight='bold', fontsize=13, color=SLATE)
    ax.set_title('Model Accuracy Ranking', fontsize=18, fontweight='bold', color=SLATE, pad=15)
    ax.tick_params(colors=SLATE)

    # put the exact number next to each bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{w:.4f}', va='center', fontweight='bold', fontsize=11, color=SLATE)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/02_model_ranking_horizontal.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm_dict, best_name):
    """
    Heatmap-style confusion matrix for the best-performing model.
    Using a teal-based colormap to stay consistent with the palette.
    """
    print("  → Confusion matrix for best model")
    cm = cm_dict[best_name]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # custom teal-ish colormap
    cmap = sns.light_palette(DARK_TEAL, as_cmap=True)

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Sports', 'Politics'],
                yticklabels=['Sports', 'Politics'],
                annot_kws={'size': 20, 'weight': 'bold', 'color': DEEP_NAVY},
                linewidths=2, linecolor='white',
                ax=ax)

    ax.set_title(f'Confusion Matrix\n{best_name}', fontsize=15, fontweight='bold', color=SLATE, pad=15)
    ax.set_ylabel('True Label', fontweight='bold', color=SLATE)
    ax.set_xlabel('Predicted Label', fontweight='bold', color=SLATE)
    ax.tick_params(colors=SLATE)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/03_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar(df):
    """
    Small radar/spider chart showing accuracy & F1 for the best model.
    A bit redundant info-wise, but it looks good in the report.
    """
    print("  → Radar performance chart")
    best = df.sort_values('Accuracy', ascending=False).iloc[0]

    cats  = ['Accuracy', 'F1-Score', 'Accuracy', 'F1-Score']
    vals  = [best['Accuracy'], best['F1-Score'], best['Accuracy'], best['F1-Score']]

    N = len(cats)
    angles = [n / N * 2 * pi for n in range(N)]
    vals  += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, vals, linewidth=2.5, color=CORAL)
    ax.fill(angles, vals, color=CORAL, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'F1', 'Accuracy', 'F1'],
                       fontsize=13, fontweight='bold', color=SLATE)
    ax.set_rlabel_position(30)
    ax.set_yticks([0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.90', '0.95', '1.00'], color='grey', fontsize=10)
    ax.set_ylim(0.85, 1.0)
    ax.spines['polar'].set_color('#CCCCCC')
    ax.grid(color='#CCCCCC', alpha=0.4)

    ax.set_title(f"Performance Profile\n{best['Configuration']}",
                 fontsize=16, fontweight='bold', color=SLATE, pad=25)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/04_radar_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


# ---------- grouped bar chart (bonus — accuracy vs F1 side by side) ----------
def plot_grouped_bars(df):
    """
    Side-by-side bar chart comparing accuracy and F1 for each config.
    Added this because I thought it gives a cleaner comparison than
    the ranking chart alone — you can see at a glance where accuracy
    and F1 diverge.
    """
    print("  → Grouped accuracy vs F1 chart")
    df_sorted = df.sort_values('Accuracy', ascending=False)

    x = np.arange(len(df_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width / 2, df_sorted['Accuracy'],  width, label='Accuracy', color=TEAL, edgecolor='white')
    bars2 = ax.bar(x + width / 2, df_sorted['F1-Score'],  width, label='F1-Score', color=CORAL, edgecolor='white')

    ax.set_ylabel('Score', fontweight='bold', fontsize=13, color=SLATE)
    ax.set_title('Accuracy vs F1-Score by Configuration', fontsize=17, fontweight='bold', color=SLATE, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Configuration'], rotation=40, ha='right', fontsize=10, color=SLATE)
    ax.legend(fontsize=12)
    ax.set_ylim(0.60, 1.05)
    ax.tick_params(colors=SLATE)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/05_grouped_acc_f1.png', dpi=300, bbox_inches='tight')
    plt.close()


# ========== MAIN ==========
if __name__ == '__main__':

    # 1. load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. run all experiments
    results_df, cm_dict = run_experiments(X_train, X_test, y_train, y_test)

    # 3. save results CSV
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv(f'{SAVE_DIR}/final_results.csv', index=False)
    print('\nResults (sorted by accuracy):')
    print(results_df.to_string(index=False))

    # 4. generate all the charts
    print('\nGenerating visualisations...')
    plot_top_keywords(X_train, y_train, 'Sports')
    plot_top_keywords(X_train, y_train, 'Politics')
    plot_donut_chart(y)
    plot_accuracy_ranking(results_df)

    best_model = results_df.iloc[0]['Configuration']
    plot_confusion_matrix(cm_dict, best_model)
    plot_radar(results_df)
    plot_grouped_bars(results_df)

    print(f"\nDone! All figures saved in '{SAVE_DIR}/'.")

    # 5. quick interactive classification using the best model
    print('\n--- Interactive mode (type "exit" to quit) ---')
    clfs, feats, _ = get_configs()
    feat_key, clf_key = best_model.split(' + ', 1)

    best_vec = feats[feat_key]
    best_clf = clfs[clf_key]

    # re-fit on everything for the interactive demo
    X_all_vec = best_vec.fit_transform(X)
    best_clf.fit(X_all_vec, y)

    while True:
        try:
            text = input('\nPaste a document (or "exit"): ').strip()
            if text.lower() == 'exit':
                break
            if not text:
                print('Please enter some text.')
                continue
            pred = best_clf.predict(best_vec.transform([text]))
            print(f'Predicted category: {pred[0]}')
        except (KeyboardInterrupt, EOFError):
            print('\nBye!')
            break