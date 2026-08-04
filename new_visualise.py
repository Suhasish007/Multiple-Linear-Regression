import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.multicomp import MultiComparison
import networkx as nx
import os
import textwrap

# ── Fashion-editorial visual theme ──────────────────────────────────────────
# Inspired by design studio report aesthetics: structured, colourful, confident
plt.rcParams.update({
    'font.family':          'DejaVu Serif',   # editorial serif feel
    'font.size':            10,
    'axes.titlesize':       13,
    'axes.titleweight':     'bold',
    'axes.labelsize':       11,
    'axes.labelweight':     'bold',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.spines.left':     True,
    'axes.spines.bottom':   True,
    'axes.linewidth':       1.4,
    'axes.facecolor':       '#FAFAF7',       # warm off-white panel background
    'figure.facecolor':     '#F4F1EC',       # warm parchment canvas
    'figure.titlesize':     15,
    'figure.titleweight':   'bold',
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'legend.fontsize':      9,
    'legend.framealpha':    0.92,
    'legend.edgecolor':     '#CCCCCC',
    'grid.color':           '#E0DDD8',
    'grid.linewidth':       0.8,
    'axes.grid':            True,
    'grid.alpha':           0.6,
})

# Create output directories
Path("figures").mkdir(exist_ok=True)
Path("visualization_results").mkdir(exist_ok=True)

# Load the processed dataset
try:
    data = pd.read_csv('enhanced_mlr_industry_survey_results_processed.csv')
    if "Original_Q7" in data.columns:
        data.rename(columns={"Original_Q7": "Q7"}, inplace=True)
    print("Successfully loaded processed dataset.")
except FileNotFoundError:
    print("Processed dataset not found. Loading clean dataset instead.")
    try:
        data = pd.read_csv('mlr_industry_survey_results_clean.csv')
    except FileNotFoundError:
        print("No datasets found. Creating example data for demonstration.")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 200
        data = pd.DataFrame({
            'Q5': np.random.randint(1, 5, size=n_samples),
            'Q8': np.random.randint(1, 5, size=n_samples),
            'Q10': np.random.randint(1, 5, size=n_samples),
            'Q11': np.random.randint(1, 6, size=n_samples),
            'Q12': np.random.randint(1, 5, size=n_samples),
            'Q13': np.random.randint(1, 5, size=n_samples),
            'Q14_1': np.random.randint(1, 5, size=n_samples),
            'Q14_2': np.random.randint(1, 5, size=n_samples),
            'Q14_3': np.random.randint(1, 5, size=n_samples),
            'Q15': np.random.randint(1, 5, size=n_samples),
            'Q16': np.random.randint(1, 5, size=n_samples),
            'Q17': np.random.randint(1, 5, size=n_samples),
            'Q9_1': np.random.randint(1, 5, size=n_samples),
            'Q9_2': np.random.randint(1, 5, size=n_samples),
        })

# Create copy of the data for cleaning and processing
data_clean = data.copy()
data_processed = data.copy()

# ── Colour system – fashion-studio palette ───────────────────────────────────
# Inspired by runway colour stories: structured, multi-hued, memorable
FASHION_COLORS = [
    "#B5294E",   # deep crimson (Pathway 1 anchor)
    "#E07B54",   # burnt sienna
    "#E8B84B",   # warm gold
    "#4A7C59",   # forest sage (Pathway 2 anchor)
    "#3B6B8A",   # steel blue
    "#5C4A8F",   # violet (Pathway 3 anchor)
    "#C4627A",   # dusty rose
    "#2E8B8B",   # teal
]

main_palette        = sns.color_palette(FASHION_COLORS)
categorical_palette = sns.color_palette(FASHION_COLORS)

# Diverging correlation colormap: blush-pink → off-white → deep indigo
correlation_palette = LinearSegmentedColormap.from_list(
    "fashion_corr",
    ["#B5294E", "#D4849A", "#F4F1EC", "#8BA7CC", "#3B6B8A"]
)

# Per-pathway accent colours (used for network nodes & pathway headers)
PATHWAY_ACCENT = {
    "Pathway 1": "#B5294E",   # crimson  – confidence
    "Pathway 2": "#4A7C59",   # sage     – challenges
    "Pathway 3": "#5C4A8F",   # violet   – investment
}

# Mapping dictionaries for variable interpretation
q_mappings = {
    'Q7': {
        1: 'Prototyping Garments',
        2: 'AR Filters on Social',
        3: 'Metaverse Avatars',
        4: 'Online Shopping',
        5: 'Exploring Trends',
        6: 'Others'
    },
    'Q5': {
        1: 'Expert (10+ years)',
        2: 'Advanced (8-10 years)',
        3: 'Intermediate (5-7 years)',
        4: 'Mid Level (2-4 year)',
        5: 'Beginner'
    },
    'Q8': {
        1: 'Not proficient',
        2: 'Limited Proficiency',
        3: 'Somewhat proficient',
        4: 'Proficient',
        5: 'Very proficient'
    },
    'Q10': {
        5: 'Extremely confident',
        4: 'Very confident',
        3: 'Moderately confident',
        2: 'Slightly confident',
        1: 'Still unsure'
    },
    'Q11': {
        1: 'Complexity working in 3D',
        2: 'Computer hardware requirements',
        3: 'System crash',
        4: 'Lack of tutorials',
        5: 'Subscription cost',
        6: 'Others'
    },
    'Q12': {
        1: 'Complex',
        2: 'Confusing',
        3: 'Neutral',
        4: 'Manageable',
        5: 'Empowering'
    },
    'Q13': {
        5: 'Very knowledgeable',
        4: 'Knowledgeable',
        3: 'Somewhat knowledgeable',
        2: 'Limited knowledge',
        1: 'No knowledge'
    },
    'Q15': {
        5: 'Yes (highly willing)',
        4: 'Probably',
        3: 'Undecided',
        2: 'No (prefer free/cheaper)',
        1: 'Not sure'
    },
    'Q16': {
        5: '$200+',
        4: '$101-$200',
        3: '$51-$100',
        2: '$21-$50',
        1: '$0-$20'
    },
    'Q17': {
        5: 'Highly essential',
        4: 'Fairly important',
        3: 'Moderately necessary',
        2: 'Slightly important',
        1: 'Not necessary'
    }
}

# Common q9 and q14 mappings for all subcomponents
for i in range(1, 5):
    q_mappings[f'Q9_{i}'] = {
        1: 'Not useful',
        2: 'Slightly helpful',
        3: 'Somewhat beneficial',
        4: 'Very helpful',
        5: 'Incredibly beneficial'
    }

for i in range(1, 5):
    q_mappings[f'Q14_{i}'] = {
        1: 'Lacking',
        2: 'Limited',
        3: 'Moderate',
        4: 'Sufficient',
        5: 'Abundant'
    }

# Define pathways for visualization consistency
pathways = {
    'Pathway 1': {
        'title': 'Pathway 1: Factors Influencing Confidence with Digital Fashion Technologies',
        'dependent': 'Q10',
        'independent': ['Q8', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5', 'Q7_6', 'Q9_1', 'Q9_2'],
        'dependent_label': 'Confidence with Digital Fashion Technologies',
        'independent_labels': {
            'Q8': 'Proficiency Level',
            'Q7_2': 'AR Filters',
            'Q7_3': 'Metaverse Avatars',
            'Q7_4': 'Online Shopping',
            'Q7_5': 'Exploring Trends',
            'Q7_6': 'Others',
            'Q9_1': 'YouTube Tutorial Usefulness',
            'Q9_2': 'Webinar Usefulness'
        },
        'color': FASHION_COLORS[0]   # crimson
    },
    'Pathway 2': {
        'title': 'Pathway 2: Factors Influencing Challenges with 3D Virtual Fashion Design',
        'dependent': 'Q11',
        'independent': ['Q12', 'Q13', 'Q14_1', 'Q14_2'],
        'dependent_label': 'Challenges with 3D Virtual Fashion Design',
        'independent_labels': {
            'Q12': 'Software Complexity',
            'Q13': 'Teacher Knowledge',
            'Q14_1': 'Technical Resources',
            'Q14_2': 'Equipment'
        },
        'color': FASHION_COLORS[3]   # forest sage
    },
    'Pathway 3': {
        'title': 'Pathway 3: Factors Influencing Willingness to Invest in Digital Fashion Software',
        'dependent': 'Q15',
        'independent': ['Q16', 'Q17', 'Q14_1', 'Q14_3'],
        'dependent_label': 'Willingness to Invest in Software',
        'independent_labels': {
            'Q16': 'Training Payment Willingness',
            'Q17': 'Perceived Importance',
            'Q14_1': 'Technical Resources',
            'Q14_3': 'Support Resources'
        },
        'color': FASHION_COLORS[5]   # violet
    }
}

# Create text data with categorical labels
text_data = data_clean.copy()
for col in q_mappings:
    if col in text_data.columns:
        text_data[f"{col}_text"] = text_data[col].map(q_mappings[col])


# Define functions for creating visualizations
def save_figure(fig, filename, dpi=300):
    """Save figure in PNG format for publication"""
    # Create directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Save as PNG (no PDF as requested)
    fig.savefig(f'figures/{filename}.png', dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: figures/{filename}.png")


def create_distribution_plot(data, column, title, x_label, filename_suffix=""):
    """Create publication-quality distribution plot with appropriate statistics"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#F4F1EC')
    ax.set_facecolor('#FAFAF7')

    if column in q_mappings:
        value_counts = data[column].value_counts().sort_index()
        sorted_indices = value_counts.index.tolist()
        text_labels = [q_mappings[column].get(idx, str(idx)) for idx in sorted_indices]
        n_bars = len(value_counts)

        # Each bar gets its own fashion colour, cycling through the palette
        bar_colors = [FASHION_COLORS[i % len(FASHION_COLORS)] for i in range(n_bars)]
        bars = ax.bar(
            range(n_bars), value_counts.values,
            color=bar_colors, width=0.65,
            edgecolor='white', linewidth=1.2, zorder=3
        )
        ax.set_xticks(range(n_bars))
        ax.set_xticklabels(text_labels, rotation=40, ha='right', fontsize=8.5)
        plt.subplots_adjust(bottom=0.25)

        # Percentage labels on top of each bar
        total = value_counts.values.sum()
        for bar, val in zip(bars, value_counts.values):
            pct = 100 * val / total
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.3,
                f'{int(val)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=7.5,
                color='#333333', fontweight='bold'
            )

        mean_val   = data[column].mean()
        median_val = data[column].median()
        ax.axvline(mean_val - 1,   color='#B5294E', linestyle='--', lw=1.8,
                   alpha=0.85, label=f'Mean: {mean_val:.2f}', zorder=4)
        ax.axvline(median_val - 1, color='#3B6B8A', linestyle=':',  lw=1.8,
                   alpha=0.85, label=f'Median: {median_val:.2f}', zorder=4)
    else:
        sns.histplot(data[column].dropna(), kde=True, ax=ax,
                     color=FASHION_COLORS[0], bins=20,
                     edgecolor='white', linewidth=0.6, alpha=0.85)
        mean_val   = data[column].mean()
        median_val = data[column].median()
        ax.axvline(mean_val,   color='#B5294E', linestyle='--', lw=1.8,
                   alpha=0.85, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#3B6B8A', linestyle=':',  lw=1.8,
                   alpha=0.85, label=f'Median: {median_val:.2f}')

    # Stats box with refined styling
    stats_text = (f"n = {data[column].count()}\n"
                  f"Mean = {data[column].mean():.2f}\n"
                  f"SD = {data[column].std():.2f}\n"
                  f"Median = {data[column].median():.2f}")
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#CCCCCC', alpha=0.92, linewidth=1)
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')

    ax.set_title(title, pad=12, color='#222222')
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel('Frequency', labelpad=8)
    ax.legend(loc='upper left', framealpha=0.92)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_figure(fig, f"distribution_{column}{filename_suffix}")
    plt.close()


def create_correlation_heatmap(data, columns, title, filename_suffix=""):
    """Create publication-quality correlation heatmap"""
    corr_matrix = data[columns].corr(method='spearman', numeric_only=True)
    mask = np.zeros_like(corr_matrix, dtype=bool)

    # Dynamic figure sizing and prevent clutter
    fig_width = max(12, len(columns) * 1.2)
    fig_height = max(10, len(columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#F4F1EC')
    ax.set_facecolor('#FAFAF7')

    var_labels = {}
    for col in columns:
        for pathway in pathways.values():
            if col in pathway['independent_labels']:
                var_labels[col] = pathway['independent_labels'][col]
            elif col == pathway['dependent']:
                var_labels[col] = pathway['dependent_label']
        if col not in var_labels:
            var_labels[col] = col

    # Create custom annotations to completely prevent overlapping text
    annot_array = np.empty_like(corr_matrix, dtype=object)
    for i in range(len(columns)):
        for j in range(len(columns)):
            if mask[i, j]:
                annot_array[i, j] = ""
            else:
                corr = corr_matrix.iloc[i, j]
                x_data = data[columns[i]].dropna()
                y_data = data[columns[j]].dropna()
                common_index = x_data.index.intersection(y_data.index)
                if len(common_index) > 2:
                    p_value = stats.spearmanr(data.loc[common_index, columns[i]], data.loc[common_index, columns[j]])[1]
                    sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
                    annot_array[i, j] = f"{corr:.2f}{sig}"
                else:
                    annot_array[i, j] = f"{corr:.2f}"

    annot_size = 10 if len(columns) <= 6 else (9 if len(columns) <= 10 else 8)
    
    wrapped_labels = [textwrap.fill(var_labels[col], width=25) for col in columns]

    heatmap = sns.heatmap(
        corr_matrix, mask=mask, annot=annot_array, fmt="",
        cmap=correlation_palette, square=True, linewidths=1.0,
        linecolor='#F4F1EC', ax=ax,
        vmin=-1, vmax=1,
        annot_kws={"size": annot_size, "weight": "bold"},
        cbar_kws={"shrink": 0.75, "label": "Pearson's r",
                  "ticks": [-1, -0.5, 0, 0.5, 1]},
        xticklabels=wrapped_labels,
        yticklabels=wrapped_labels
    )
    
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=40, ha='right', fontsize=annot_size)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, va='center', fontsize=annot_size)

    # Style the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Spearman's ρ", fontsize=9, labelpad=8)


    # Styled title bar
    ax.set_title(title, pad=16, fontsize=14, fontweight='bold', color='#1A1A1A')

    fig.text(0.01, 0.02, "NOTE: Spearman's ρ used for correlations (appropriate for ordinal survey data).",
             fontsize=7.5, color='#666666', style='italic')
    fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001",
             fontsize=9, color='#555555', style='italic')

    # Removed tight_layout to allow bottom/left manual adjustments to work
    plt.subplots_adjust(bottom=0.35, left=0.35, right=0.95, top=0.9)
    save_figure(fig, f"correlation_heatmap{filename_suffix}")
    plt.close()


def create_regression_plot(data, x, y, x_label, y_label, title, filename_suffix=""):
    """Create regression plot between two variables with appropriate treatment of ordinal data.
    
    NOTE: For ordinal/categorical predictors and outcomes (Likert scales, nominal data),
    points are jittered and OLS regression line is presented for descriptive purposes only.
    Ordinal regression (OrderedModel) or logistic regression should be preferred for
    statistical inference on such data."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('#F4F1EC')
    ax.set_facecolor('#FAFAF7')

    # Determine accent colour from pathway membership
    scatter_color = FASHION_COLORS[0]
    for pid, pathway in pathways.items():
        if x in pathway['independent'] or x == pathway['dependent']:
            scatter_color = pathway['color']
            break

    is_ordinal_pair = (x in q_mappings and y in q_mappings)
    
    # Check if either variable is explicitly nominal
    is_nominal_x = (x in ['Q7', 'Q11'])
    
    if is_nominal_x:
        # Create barplot with SE and underlying stripplot for nominal predictors
        sns.barplot(x=x, y=y, data=data, ax=ax,
                    capsize=0.1, errorbar='se', 
                    color=scatter_color, alpha=0.7)
        sns.stripplot(x=x, y=y, data=data, ax=ax,
                      color='black', alpha=0.3, jitter=0.2, size=4)
        # Note: Do not draw an OLS line for nominal categories
        
    elif is_ordinal_pair:
        # For ordinal data: use jittered scatter plot (OLS line is descriptive only)
        jitter_strength = 0.15
        x_jittered = data[x] + np.random.normal(0, jitter_strength, size=len(data))
        y_jittered = data[y] + np.random.normal(0, jitter_strength, size=len(data))
        ax.scatter(x_jittered, y_jittered, alpha=0.45, s=55, color=scatter_color,
                  edgecolors='white', linewidths=0.5)
        
        # Add OLS line for descriptive context only
        z = np.polyfit(data[x].dropna(), data[y].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[x].min(), data[x].max(), 100)
        ax.plot(x_line, p(x_line), color='#B5294E', lw=2.2, alpha=0.9, linestyle='--',
               label='OLS line (descriptive)')
    else:
        # For continuous or mixed data: use standard regplot
        sns.regplot(
            x=x, y=y, data=data,
            scatter_kws={'alpha': 0.55, 's': 55, 'color': scatter_color,
                         'edgecolors': 'white', 'linewidths': 0.5},
            line_kws={'color': '#B5294E', 'lw': 2.2, 'alpha': 0.9},
            ci=95, ax=ax
        )

    if is_nominal_x:
        groups = [group.dropna() for name, group in data.groupby(x)[y]]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            sig_text = ('***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '')
            stats_text = f"ANOVA F = {f_stat:.2f}{sig_text}\np = {p_value:.4f}\n(Difference across categories)"
        else:
            stats_text = "Insufficient Data"
    else:
        corr, p_value = stats.pearsonr(data[x].dropna(), data[y].dropna()) if not is_ordinal_pair else stats.spearmanr(data[x].dropna(), data[y].dropna())
        sig_text = ('***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '')

        # Refined annotation box
        if is_ordinal_pair:
            stats_text = f"ρ = {corr:.3f}{sig_text}\np = {p_value:.4f}\n(Spearman's ρ on ordinal data)"
        else:
            stats_text = f"r = {corr:.3f}{sig_text}\np = {p_value:.4f}"

    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#CCCCCC', alpha=0.95, linewidth=1.2)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='#222222')

    if x in q_mappings:
        x_values = sorted(data[x].unique())
        x_labels = [q_mappings[x].get(val, str(val)) for val in x_values]
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=40, ha='right')

    if y in q_mappings:
        y_values = sorted(data[y].unique())
        y_labels = [q_mappings[y].get(val, str(val)) for val in y_values]
        ax.set_yticks(y_values)
        ax.set_yticklabels(y_labels)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')

    wrapped_x_label = textwrap.fill(x_label, width=30)
    wrapped_y_label = textwrap.fill(y_label, width=30)
    ax.set_xlabel(wrapped_x_label, fontsize=11, labelpad=10)
    ax.set_ylabel(wrapped_y_label, fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14, color='#1A1A1A')
    
    if is_ordinal_pair:
        fig.text(0.01, 0.025, "NOTE: Ordinal data displayed with jitter. Points are randomly displaced for visibility. OLS line shown for descriptive context only; ordinal/ordered logistic regression recommended for inference.",
                 fontsize=7.5, color='#B5294E', style='italic', wrap=True)
        fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=8, color='#666666', style='italic')
    else:
        fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=8, color='#666666', style='italic')
    plt.tight_layout()
    save_figure(fig, f"regression_{x}_{y}{filename_suffix}")
    plt.close()

def create_multi_regression_plot(data, x_vars, y_var, title, filename_suffix=""):
    """Create multi-panel regression plot with one panel per independent variable.

    NOTE: For ordinal/categorical variables, jittered points are displayed and OLS
    regression lines are shown for descriptive purposes only. Ordinal regression
    (OrderedModel) should be preferred for statistical inference."""
    n_vars = len(x_vars)

    # Create figure with fixed subplot parameters
    fig = plt.figure(figsize=(15, 6.0 * ((n_vars + 1) // 2)))

    # Use more space between plots
    gs = gridspec.GridSpec(((n_vars + 1) // 2), 2, hspace=0.8, wspace=0.4)

    # Get dependent variable label
    y_label = ""
    for pathway in pathways.values():
        if y_var == pathway['dependent']:
            y_label = pathway['dependent_label']
    if not y_label:
        y_label = y_var

    # Check if dependent variable is ordinal
    y_is_ordinal = y_var in q_mappings

    # Create subplot for each independent variable
    for i, x_var in enumerate(x_vars):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        # Get independent variable label
        x_label = ""
        for pathway in pathways.values():
            if x_var in pathway['independent_labels']:
                x_label = pathway['independent_labels'][x_var]
        if not x_label:
            x_label = x_var

        # Check if both variables are ordinal
        is_ordinal_pair = (x_var in q_mappings and y_is_ordinal)

        # Create scatter plot with regression line
        if is_ordinal_pair:
            jitter_strength = 0.12
            x_jittered = data[x_var] + np.random.normal(0, jitter_strength, size=len(data))
            y_jittered = data[y_var] + np.random.normal(0, jitter_strength, size=len(data))
            ax.scatter(x_jittered, y_jittered, alpha=0.4, s=35,
                      color=main_palette[i % len(main_palette)], edgecolors='white', linewidths=0.3)

            # Add OLS line for descriptive context
            z = np.polyfit(data[x_var].dropna(), data[y_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_var].min(), data[x_var].max(), 100)
            ax.plot(x_line, p(x_line), color='red', lw=1.5, alpha=0.8, linestyle='--')
        else:
            # For continuous or mixed data: use standard regplot
            sns.regplot(
                x=x_var, y=y_var, data=data,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': main_palette[i % len(main_palette)]},
                line_kws={'color': 'red', 'lw': 1.5},
                ci=95,  # 95% confidence interval
                ax=ax
            )

            # Calculate and display correlation
            # Initialise with NaN so p_value is always bound even if computation fails
            # (e.g. binary dummy variables with near-zero variance can cause exceptions)
            corr = np.nan
            p_value = np.nan

            try:
                if is_ordinal_pair:
                    corr, p_value = stats.spearmanr(
                        data[x_var].dropna(), data[y_var].dropna()
                    )
                else:
                    # For binary dummies (Q7_2, Q7_3 etc.) pearsonr can fail on
                    # near-constant columns — catch and fall back to spearman
                    x_vals = data[x_var].dropna()
                    y_vals = data[y_var].dropna()
                    if x_vals.nunique() <= 2:
                        corr, p_value = stats.spearmanr(x_vals, y_vals)
                    else:
                        corr, p_value = stats.pearsonr(x_vals, y_vals)
            except Exception:
                corr = np.nan
                p_value = np.nan

            # Add significance stars
            sig_text = ""
            if np.isfinite(p_value):
                if p_value < 0.001:
                    sig_text = "***"
                elif p_value < 0.01:
                    sig_text = "**"
                elif p_value < 0.05:
                    sig_text = "*"

            # Add correlation annotation
            if np.isfinite(corr):
                if is_ordinal_pair or (data[x_var].nunique() <= 2):
                    stats_text = f"ρ = {corr:.3f}{sig_text}\n(Spearman's ρ)"
                else:
                    stats_text = f"r = {corr:.3f}{sig_text}"
            else:
                stats_text = "corr = N/A\n(insufficient variance)"
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        # Adjust categorical axis if needed
        if x_var in q_mappings:
            x_values = sorted(data[x_var].unique())
            x_labels = [q_mappings[x_var].get(val, str(val)) for val in x_values]

            # Set custom ticks for x-axis
            x_ticks = sorted(data[x_var].unique())
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

            # Adjust x-limits to ensure labels are visible
            margin = 0.2
            ax.set_xlim(min(x_ticks) - margin, max(x_ticks) + margin)

        if y_var in q_mappings:
            y_values = sorted(data[y_var].unique())
            y_labels = [q_mappings[y_var].get(val, str(val)) for val in y_values]

            # Set custom ticks for y-axis
            y_ticks = sorted(data[y_var].unique())
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=7)

            # Adjust y-limits to ensure labels are visible
            margin = 0.2
            ax.set_ylim(min(y_ticks) - margin, max(y_ticks) + margin)

        # Shorter wrapped text for labels to avoid overflow
        wrapped_x_label = textwrap.fill(x_label, width=18)
        ax.set_xlabel(wrapped_x_label, fontsize=8, labelpad=10)

        # Only add y-label on left plots
        if i % 2 == 0:
            wrapped_y_label = textwrap.fill(y_label, width=18)
            ax.set_ylabel(wrapped_y_label, fontsize=8, labelpad=10)
        else:
            ax.set_ylabel('')

    # Add title
    plt.suptitle(title, fontsize=14, y=0.98)

    # Add significance note only where relevant
    if y_is_ordinal:
        plt.figtext(0.01, 0.02, "NOTE: Ordinal outcome displayed with jitter for visibility. OLS lines shown for descriptive context only; ordinal regression recommended for inference.",
                   fontsize=7.5, color='#B5294E', style='italic')
        plt.figtext(0.01, 0.01, "Correlation significance: * p<0.05, ** p<0.01, *** p<0.001", fontsize=8)
    else:
        plt.figtext(0.01, 0.01, "Correlation significance: * p<0.05, ** p<0.01, *** p<0.001", fontsize=8)

    # Use fixed subplot parameters instead of tight_layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.6, wspace=0.4)

    save_figure(fig, f"multi_regression_{y_var}{filename_suffix}")
    plt.close()

def create_coefficient_plot(model_results, title, filename_suffix=""):
    """Create coefficient plot with confidence intervals for regression model"""
    params   = model_results.params
    conf_int = model_results.conf_int()
    intercept_name = params.index[0]

    coefs    = params.drop(intercept_name)
    conf_int = conf_int.drop(intercept_name)
    indices  = np.argsort(abs(coefs))
    coefs    = coefs.iloc[indices]
    conf_int = conf_int.iloc[indices, :]

    var_labels = {}
    for col in coefs.index:
        for pathway in pathways.values():
            if col in pathway['independent_labels']:
                var_labels[col] = pathway['independent_labels'][col]
        if col not in var_labels:
            var_labels[col] = col

    n = len(coefs)
    fig, ax = plt.subplots(figsize=(9, 1.2 + n * 0.75))
    fig.patch.set_facecolor('#F4F1EC')
    ax.set_facecolor('#FAFAF7')

    y_pos = np.arange(n)

    # Alternating light bands for readability
    for i in range(n):
        band_col = '#EFEFEB' if i % 2 == 0 else '#FAFAF7'
        ax.axhspan(i - 0.45, i + 0.45, color=band_col, zorder=0)

    # Assign a unique fashion colour to each predictor
    dot_colors = [FASHION_COLORS[i % len(FASHION_COLORS)] for i in range(n)]

    for i, (coef, color) in enumerate(zip(coefs, dot_colors)):
        err_lo = coef - conf_int.iloc[i, 0]
        err_hi = conf_int.iloc[i, 1] - coef
        ax.errorbar(
            x=coef, y=i,
            xerr=np.array([[err_lo], [err_hi]]),
            fmt='o', capsize=5, elinewidth=2, markeredgewidth=1.5,
            markersize=9, color=color, ecolor=color, markeredgecolor='white',
            zorder=5
        )

    ax.axvline(x=0, color='#888888', linestyle='-', lw=1.2, alpha=0.7, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([var_labels[col] for col in coefs.index], fontsize=10)
    ax.set_xlabel('Effect Size (Coefficient Value) with 95% Confidence Interval', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=14, color='#1A1A1A', fontweight='bold')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')

    # Significance stars indicating hypothesis test results
    p_values = model_results.pvalues.drop(intercept_name).iloc[indices]
    for i, (coef, p_val) in enumerate(zip(coefs, p_values)):
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else '')
        if sig:
            # Significant path label
            ax.text(coef + 0.003 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    i, f' {sig}', va='center', fontsize=10,
                    color='#B5294E', fontweight='bold')

    stats_text = (f"Model Hypothesis F-Test:\n"
                  f"R² = {model_results.rsquared:.3f}, Adj. R² = {model_results.rsquared_adj:.3f}\n"
                  f"F({int(model_results.df_model)}, {int(model_results.df_resid)}) = {model_results.fvalue:.2f}, p = {model_results.f_pvalue:.4f}")
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor='#CCCCCC', alpha=0.92, linewidth=1)
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    fig.text(0.01, 0.01, "Hypothesis Significance:\n* p<0.05 (Supported)  ** p<0.01 (Supported)  *** p<0.001 (Strongly Supported)\nNOTE: Displayed coefficients are from OLS models. These serve as a robustness check; primary outcomes for ordinal variables utilize ordered logistic regression.",
             fontsize=8, color='#666666', style='italic')
    plt.tight_layout()
    save_figure(fig, f"coefficients_{filename_suffix}")
    plt.close()


def create_residual_plots(model_results, title_prefix, filename_suffix=""):
    """Create diagnostic plots for regression model residuals"""
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#F4F1EC')
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2, ax3, ax4]

    for ax in axes:
        ax.set_facecolor('#FAFAF7')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#BBBBBB')
        ax.spines['bottom'].set_color('#BBBBBB')

    # Assign one fashion colour per panel
    PANEL_COLORS = [FASHION_COLORS[0], FASHION_COLORS[3],
                    FASHION_COLORS[5], FASHION_COLORS[7 % len(FASHION_COLORS)]]

    y_pred       = model_results.fittedvalues
    resid        = model_results.resid
    student_resid = model_results.get_influence().resid_studentized_internal

    short_title = title_prefix[:37] + "..." if len(title_prefix) > 40 else title_prefix

    # 1. Residuals vs Fitted
    axes[0].scatter(y_pred, resid, color=PANEL_COLORS[0],
                    alpha=0.55, s=50, edgecolors='white', linewidths=0.4, zorder=3)
    axes[0].axhline(y=0, color='#B5294E', linestyle='-', lw=1.4)
    lowess = sm.nonparametric.lowess(resid, y_pred, frac=0.6)
    axes[0].plot(lowess[:, 0], lowess[:, 1], color='#E07B54', linestyle='--', lw=1.8)
    axes[0].set_title("Residuals vs Fitted", fontsize=12, pad=10, fontweight='bold')
    axes[0].set_xlabel("Fitted values", fontsize=10, labelpad=8)
    axes[0].set_ylabel("Residuals", fontsize=10, labelpad=8)

    # 2. Normal Q-Q
    theoretical_quantiles = stats.probplot(student_resid, dist="norm")[0][0]
    sorted_residuals = np.sort(student_resid)
    axes[1].scatter(theoretical_quantiles, sorted_residuals,
                    color=PANEL_COLORS[1], alpha=0.55, s=50,
                    edgecolors='white', linewidths=0.4, zorder=3)
    mn = min(theoretical_quantiles.min(), sorted_residuals.min())
    mx = max(theoretical_quantiles.max(), sorted_residuals.max())
    axes[1].plot([mn, mx], [mn, mx], color='#B5294E', lw=1.6)
    axes[1].set_title("Normal Q-Q", fontsize=12, pad=10, fontweight='bold')
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=10, labelpad=8)
    axes[1].set_ylabel("Standardized Residuals", fontsize=10, labelpad=8)

    # 3. Scale-Location
    axes[2].scatter(y_pred, np.sqrt(np.abs(student_resid)),
                    color=PANEL_COLORS[2], alpha=0.55, s=50,
                    edgecolors='white', linewidths=0.4, zorder=3)
    lowess2 = sm.nonparametric.lowess(np.sqrt(np.abs(student_resid)), y_pred, frac=0.6)
    axes[2].plot(lowess2[:, 0], lowess2[:, 1], color='#E8B84B', linestyle='--', lw=1.8)
    axes[2].set_title("Scale-Location", fontsize=12, pad=10, fontweight='bold')
    axes[2].set_xlabel("Fitted values", fontsize=10, labelpad=8)
    axes[2].set_ylabel("√|Standardized Residuals|", fontsize=10, labelpad=8)

    # 4. Residuals vs Leverage
    influence = model_results.get_influence()
    leverage  = influence.hat_matrix_diag
    cooks_d   = influence.cooks_distance[0]
    min_lev   = max(0.001, min(leverage))
    lev_range = np.linspace(min_lev, max(leverage) * 1.05, 100)

    scatter = axes[3].scatter(
        leverage, student_resid,
        c=cooks_d, cmap='RdYlGn_r', alpha=0.65,
        s=np.clip(cooks_d * 2000, 20, 300),
        edgecolors='white', linewidths=0.4, zorder=3
    )
    plt.colorbar(scatter, ax=axes[3], label="Cook's Distance", fraction=0.046, pad=0.04)
    p = len(model_results.params)
    for level in [0.5, 1]:
        boundary = np.sqrt(p * level * (1 - lev_range) / lev_range)
        axes[3].plot(lev_range, boundary,  'k--', lw=1, alpha=0.6,
                     label=f"Cook's d = {level}")
        axes[3].plot(lev_range, -boundary, 'k--', lw=1, alpha=0.6)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_title("Residuals vs Leverage", fontsize=12, pad=10, fontweight='bold')
    axes[3].set_xlabel("Leverage", fontsize=10, labelpad=8)
    axes[3].set_ylabel("Standardized Residuals", fontsize=10, labelpad=8)

    fig.suptitle(short_title, fontsize=14, y=0.99, fontweight='bold', color='#1A1A1A')
    fig.text(0.01, 0.01, "NOTE: These diagnostic plots are derived from OLS models. These serve as a robustness check; primary outcomes for ordinal variables utilize ordered logistic regression.",
             fontsize=9, color='#666666', style='italic')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08,
                        hspace=0.45, wspace=0.42)
    save_figure(fig, f"residual_diagnostics_{filename_suffix}")
    plt.close()

def create_pathway_interaction_plots(data, filename_suffix=""):
    """Create visualization showing accurate pathway interactions between outcome variables.

    NOTE: For ordinal outcome variables (Likert scales), jittered points are displayed.
    OLS regression lines shown for descriptive context only; ordinal regression recommended."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor('#F4F1EC')

    interactions = [
        ('Q10', 'Q15', 'Confidence with Digital Fashion Tech', 'Willingness to Invest'),
        ('Q11', 'Q15', 'Challenges with 3D Design', 'Willingness to Invest')
    ]

    for i, (ind_var, dep_var, x_label, y_label) in enumerate(interactions):
        ax = axes[i]
        ax.set_facecolor('#FAFAF7')

        is_ordinal_pair = (ind_var in q_mappings and dep_var in q_mappings)

        if is_ordinal_pair:
            jitter_strength = 0.15
            x_jittered = data[ind_var] + np.random.normal(0, jitter_strength, size=len(data))
            y_jittered = data[dep_var] + np.random.normal(0, jitter_strength, size=len(data))
            ax.scatter(x_jittered, y_jittered, alpha=0.5, color=FASHION_COLORS[i*3],
                      edgecolors='white', linewidths=0.4)

            # Add OLS line for descriptive context
            z = np.polyfit(data[ind_var].dropna(), data[dep_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[ind_var].min(), data[ind_var].max(), 100)
            ax.plot(x_line, p(x_line), color='#B5294E', lw=2, alpha=0.8, linestyle='--')
        else:
            sns.regplot(x=ind_var, y=dep_var, data=data,
                        scatter_kws={'alpha': 0.6, 'color': FASHION_COLORS[i*3]},
                        line_kws={'color': '#B5294E'}, ax=ax)

        wrapped_x_label = textwrap.fill(x_label, width=25)
        wrapped_y_label = textwrap.fill(y_label, width=25)
        ax.set_xlabel(wrapped_x_label, fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel(wrapped_y_label, fontsize=11, fontweight='bold', labelpad=10)

        # Calculate stats
        clean_data = data[[ind_var, dep_var]].dropna()
        if len(clean_data) > 0:
            if is_ordinal_pair:
                corr, p_value = stats.spearmanr(clean_data[ind_var], clean_data[dep_var])
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                stats_text = f"ρ = {corr:.3f}{sig}\n(Spearman's ρ)"
            else:
                corr, p_value = stats.pearsonr(clean_data[ind_var], clean_data[dep_var])
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                stats_text = f"r = {corr:.3f}{sig}"
            props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC', alpha=0.9)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        apply_axis_labels(ax, ind_var, dep_var, data)

    plt.suptitle('Pathway Interactions Between Key Outcome Variables',
                 fontsize=14, y=0.98, fontweight='bold', color='#1A1A1A')
    plt.figtext(0.01, 0.02, "NOTE: Ordinal data displayed with jitter for visibility. OLS lines for descriptive context only; ordinal regression recommended for inference.",
               fontsize=7.5, color='#B5294E', style='italic')
    plt.figtext(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001", fontsize=9, color='#666666', style='italic')
    plt.subplots_adjust(bottom=0.25, top=0.88, wspace=0.3)
    save_figure(fig, f"pathway_interactions{filename_suffix}")
    plt.close()


def create_correlation_network(data, columns, title, filename_suffix=""):
    """Create network graph showing accurate correlations between variables.

    ISSUE #18 NOTE: Survey items (Q5, Q8, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q9) are ordinal variables
    (Likert scales or categorical). Network edge weights use Spearman's ρ. All results should be interpreted
    as exploratory, with ordinal regression (OrderedModel) used for formal inference."""
    corr_matrix = data[columns].corr(method='spearman', numeric_only=True)
    G = nx.DiGraph()

    var_labels = {}
    for col in columns:
        for pathway in pathways.values():
            if col in pathway['independent_labels']:
                var_labels[col] = pathway['independent_labels'][col]
            elif col == pathway['dependent']:
                var_labels[col] = pathway['dependent_label']
        if col not in var_labels:
            var_labels[col] = col

    for col in columns:
        G.add_node(col, label=textwrap.fill(var_labels[col], width=16))

    node_pathway = {}
    pathway_colors = {
        "Pathway 1": PATHWAY_ACCENT["Pathway 1"],
        "Pathway 2": PATHWAY_ACCENT["Pathway 2"],
        "Pathway 3": PATHWAY_ACCENT["Pathway 3"],
        "Other": FASHION_COLORS[7 % len(FASHION_COLORS)]
    }

    for col in columns:
        assigned = False
        for pid, pathway in pathways.items():
            if col == pathway['dependent'] or col in pathway['independent']:
                node_pathway[col] = pid
                assigned = True
                break
        if not assigned:
            node_pathway[col] = "Other"

    # Identify literal pathway rules
    valid_pathway_edges = set()
    for pid, pathway in pathways.items():
        dep = pathway['dependent']
        for ind in pathway['independent']:
            valid_pathway_edges.add((ind, dep))

    for i in range(len(columns)):
        for j in range(len(columns)):
            if i != j:
                u, v = columns[i], columns[j]
                try:
                    corr = corr_matrix.loc[u, v]
                    is_pathway = (u, v) in valid_pathway_edges
                    rev_is_pathway = (v, u) in valid_pathway_edges
                    
                    passes_filter = False
                    if abs(corr) >= 0.2:  # Meaningful side effect mapping
                        x_data = data[u].dropna()
                        y_data = data[v].dropna()
                        common_index = x_data.index.intersection(y_data.index)
                        if len(common_index) > 2:
                            p_value = stats.spearmanr(data.loc[common_index, u], data.loc[common_index, v])[1]
                            if p_value <= 0.05:
                                passes_filter = True
                                
                    if is_pathway:
                        G.add_edge(u, v, weight=abs(corr), sign=np.sign(corr), corr=corr, is_pathway=True)
                    elif passes_filter and i < j and not rev_is_pathway:
                        G.add_edge(u, v, weight=abs(corr), sign=np.sign(corr), corr=corr, is_pathway=False)
                except Exception:
                    pass

    fig_net, ax = plt.subplots(figsize=(20, 18))
    fig_net.patch.set_facecolor('#F4F1EC')
    ax.set_facecolor('#F4F1EC')

    # Expanded k factor prevents layout scrunching overlapping text labels
    pos = nx.spring_layout(G, seed=45, k=4.0)

    node_colors = [pathway_colors[node_pathway[node]] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3800, node_color=node_colors,
                           edgecolors='white', linewidths=2.5, ax=ax)

    for (u, v, d) in G.edges(data=True):
        width = d['weight'] * 6
        color = '#4A7C59' if d['sign'] > 0 else '#B5294E'
        if d.get('is_pathway', False):
            display_width = max(1.5, width)  # Ensure minimum width for pathway lines
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=display_width,
                                   edge_color=color, alpha=0.9, arrows=True, arrowsize=25, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width * 0.5,
                                   edge_color=color, alpha=0.4, style='dashed', arrows=False, ax=ax)

    label_options = {
        "bbox": {"boxstyle": "round,pad=0.5", "fc": "white",
                 "ec": "#CCCCCC", "alpha": 0.95, "linewidth": 1.5},
        "font_size": 11, "font_weight": "bold", "font_color": "#1A1A1A"
    }
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), **label_options, ax=ax)

    edge_labels = {(u, v): f"{d['corr']:.2f}" for u, v, d in G.edges(data=True) if d.get('is_pathway', False) or abs(d['corr']) >= 0.3}
    edge_label_options = {
        "font_size": 9, "font_color": "#3B6B8A",
        "bbox": {"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9, "ec": "none"},
        "horizontalalignment": "center", "verticalalignment": "center"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, **edge_label_options)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PATHWAY_ACCENT["Pathway 1"], markersize=14, label="Pathway 1: Confidence"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PATHWAY_ACCENT["Pathway 2"], markersize=14, label="Pathway 2: Challenges"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PATHWAY_ACCENT["Pathway 3"], markersize=14, label="Pathway 3: Investment"),
        plt.Line2D([0], [0], color='#4A7C59', lw=3, label="Positive Pathway Factor"),
        plt.Line2D([0], [0], color='#B5294E', lw=3, label="Negative Pathway Factor"),
        plt.Line2D([0], [0], color='#666666', lw=2, linestyle='dashed', alpha=0.5, label="Secondary Factor Interaction")
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.95, edgecolor='#CCCCCC', fontsize=11)

    # Extra padding to axes manually locks in margins preventing Node crops
    x_values, y_values = zip(*pos.values())
    x_margin = (max(x_values) - min(x_values)) * 0.25
    y_margin = (max(y_values) - min(y_values)) * 0.25

    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    ax.axis('off')

    plt.title(title, fontsize=18, pad=20, fontweight='bold', color='#1A1A1A')
    fig_net.text(0.01, 0.01, "NOTE: Spearman's ρ used for edge weights (appropriate for ordinal survey data).",
                fontsize=8, color='#666666', style='italic', wrap=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
    save_figure(fig_net, f"correlation_network{filename_suffix}")
    plt.close()

def match_labels_to_values(values, labels):
    """Ensure the number of labels matches the number of values"""
    # Handle empty cases
    if len(values) == 0 or len(labels) == 0:
        return []

    # Convert values to list if they're numpy array
    values_list = values.tolist() if hasattr(values, 'tolist') else list(values)

    # If more values than labels, pad with string versions of the values
    if len(values_list) > len(labels):
        return labels + [str(val) for val in values_list[len(labels):]]
    # If more labels than values, truncate the labels
    return labels[:len(values_list)]

import textwrap

def apply_axis_labels(ax, x_var, y_var, data, rotate_x=45, x_fontsize=8, y_fontsize=8):
    """Apply proper axis labels with consistent formatting"""
    # Handle x-axis categorical variables
    if x_var in q_mappings:
        x_values = sorted(data[x_var].dropna().unique())
        if len(x_values) > 0:  # Only proceed if we have values
            x_labels = [q_mappings[x_var].get(val, str(val)) for val in x_values]
            x_labels = match_labels_to_values(x_values, x_labels)
            x_labels = [textwrap.fill(lbl, width=15) for lbl in x_labels]
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, rotation=rotate_x, ha='right', fontsize=x_fontsize)

    # Handle y-axis categorical variables
    if y_var in q_mappings:
        y_values = sorted(data[y_var].dropna().unique())
        if len(y_values) > 0:  # Only proceed if we have values
            y_labels = [q_mappings[y_var].get(val, str(val)) for val in y_values]
            y_labels = match_labels_to_values(y_values, y_labels)
            y_labels = [textwrap.fill(lbl, width=15) for lbl in y_labels]
            ax.set_yticks(y_values)
            ax.set_yticklabels(y_labels, fontsize=y_fontsize)

def create_combined_plot(data, var_pairs, title, filename_suffix=""):
    """Create a figure with multiple correlation plots.

    NOTE: For ordinal/categorical variables, jittered points are displayed and OLS
    regression lines shown for descriptive purposes only. Ordinal regression recommended
    for statistical inference."""
    n_pairs = len(var_pairs)

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 4.5 * ((n_pairs + 1) // 2)))
    gs = gridspec.GridSpec(((n_pairs + 1) // 2), 2, hspace=0.6, wspace=0.4)  # Increased spacing

    # Create subplot for each variable pair
    for i, (x_var, y_var, x_label, y_label) in enumerate(var_pairs):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        # Check for nominal variables
        is_nominal_x = (x_var in ['Q7', 'Q11'])

        # Check if both variables are ordinal
        is_ordinal_pair = (x_var in q_mappings and y_var in q_mappings)

        # Create scatter plot or boxplot
        if is_nominal_x:
            # Use barplot with SE and underlying stripplot
            sns.barplot(x=x_var, y=y_var, data=data, ax=ax,
                        capsize=0.1, errorbar='se',
                        color=categorical_palette[i % len(categorical_palette)], alpha=0.7)
            sns.stripplot(x=x_var, y=y_var, data=data, ax=ax,
                          color='black', alpha=0.3, jitter=0.2, size=3)
        elif is_ordinal_pair:
            jitter_strength = 0.12
            x_jittered = data[x_var] + np.random.normal(0, jitter_strength, size=len(data))
            y_jittered = data[y_var] + np.random.normal(0, jitter_strength, size=len(data))
            ax.scatter(x_jittered, y_jittered, alpha=0.4, s=35,
                      color=categorical_palette[i % len(categorical_palette)],
                      edgecolors='white', linewidths=0.3)

            # Add OLS line for descriptive context
            z = np.polyfit(data[x_var].dropna(), data[y_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_var].min(), data[x_var].max(), 100)
            ax.plot(x_line, p(x_line), color='red', lw=1.5, alpha=0.8, linestyle='--')
        else:
            sns.regplot(
                x=x_var, y=y_var, data=data,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': categorical_palette[i % len(categorical_palette)]},
                line_kws={'color': 'red', 'lw': 1.5},
                ci=95,
                ax=ax
            )

        # Calculate correlation — Spearman for ordinal, Pearson for continuous
        if is_ordinal_pair:
            corr, p_value = stats.spearmanr(data[x_var].dropna(), data[y_var].dropna())
        else:
            corr, p_value = stats.pearsonr(data[x_var].dropna(), data[y_var].dropna())

        # Add significance stars
        sig_text = ""
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"

        # Add correlation annotation
        if is_ordinal_pair:
            stats_text = f"ρ = {corr:.3f}{sig_text}\n(Spearman's ρ)"
        else:
            stats_text = f"r = {corr:.3f}{sig_text}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Set axis labels with proper wrapped text for better readability
        wrapped_x_label = textwrap.fill(x_label, width=20)  # Reduced width
        wrapped_y_label = textwrap.fill(y_label, width=20)  # Reduced width
        ax.set_xlabel(wrapped_x_label, fontsize=10, labelpad=10)
        ax.set_ylabel(wrapped_y_label, fontsize=10, labelpad=10)

        # Apply consistent axis labeling
        apply_axis_labels(ax, x_var, y_var, data)

    # Add title
    plt.suptitle(title, fontsize=14, y=0.98)

    # Add significance note for correlations
    plt.figtext(0.01, 0.02, "NOTE: Ordinal data displayed with jitter for visibility. OLS lines shown for descriptive context only; ordinal regression recommended for inference.",
               fontsize=7.5, color='#B5294E', style='italic')
    plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", fontsize=8)

    # Use fixed subplot parameters
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18, hspace=0.6, wspace=0.4)

    save_figure(fig, f"combined_plot_{filename_suffix}")
    plt.close()

def create_confidence_combined_figure(data):
    """Create a figure with four confidence-related regression plots.

    NOTE: Q10 (Confidence) is an ordinal 5-point Likert variable. Jittered points
    displayed for visibility; OLS lines shown for descriptive context only."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.4)  # Increased spacing

    # Define variables for each plot
    plots = [
        ('Q7', 'Q10', 'Experience Level', 'Confidence with Digital Fashion Technologies', 0, 0),
        ('Q8', 'Q10', 'Proficiency Level', 'Confidence with Digital Fashion Technologies', 0, 1),
        ('Q9_1', 'Q10', 'YouTube Tutorial Usefulness', 'Confidence with Digital Fashion Technologies', 1, 0),
        ('Q9_2', 'Q10', 'Online Course Usefulness', 'Confidence with Digital Fashion Technologies', 1, 1)
    ]

    # Create each subplot
    for x_var, y_var, x_label, y_label, row, col in plots:
        ax = fig.add_subplot(gs[row, col])

        # Check if the predictor is explicitly nominal
        is_nominal_x = (x_var in ['Q7', 'Q11'])
        
        if is_nominal_x:
            # For nominal data: use a barplot with SE and underlying stripplot
            sns.barplot(x=x_var, y=y_var, data=data, ax=ax,
                        capsize=0.1, errorbar='se', 
                        color=categorical_palette[row * 2 + col], alpha=0.7)
            sns.stripplot(x=x_var, y=y_var, data=data, ax=ax,
                          color='black', alpha=0.3, jitter=0.2, size=3)
            
            # Calculate ANOVA
            groups = [group.dropna() for name, group in data.groupby(x_var)[y_var]]
            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups)
                sig_text = ('***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '')
                stats_text = f"ANOVA F={f_stat:.2f}{sig_text}\n(across categories)"
            else:
                stats_text = "Insufficient Data"
            
        else:
            jitter_strength = 0.12
            x_jittered = data[x_var] + np.random.normal(0, jitter_strength, size=len(data))
            y_jittered = data[y_var] + np.random.normal(0, jitter_strength, size=len(data))
            ax.scatter(x_jittered, y_jittered, alpha=0.4, s=35,
                      color=categorical_palette[row * 2 + col], edgecolors='white', linewidths=0.3)
    
            # Add OLS line for descriptive context
            z = np.polyfit(data[x_var].dropna(), data[y_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_var].min(), data[x_var].max(), 100)
            ax.plot(x_line, p(x_line), color='red', lw=1.5, alpha=0.8, linestyle='--')

            # Calculate correlation — Spearman appropriate for ordinal/Likert variables
            corr, p_value = stats.spearmanr(data[x_var].dropna(), data[y_var].dropna())

            # Add significance stars
            sig_text = ""
            if p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            elif p_value < 0.05:
                sig_text = "*"

            # Add correlation annotation
            stats_text = f"ρ = {corr:.3f}{sig_text}\n(Spearman's ρ)"
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        # Set axis labels with wrapped text
        wrapped_x_label = textwrap.fill(x_label, width=20)  # Reduced width
        wrapped_y_label = textwrap.fill(y_label, width=20)  # Reduced width
        ax.set_xlabel(wrapped_x_label, fontsize=10, labelpad=10)
        ax.set_ylabel(wrapped_y_label, fontsize=10, labelpad=10)

        # Apply consistent axis labeling
        apply_axis_labels(ax, x_var, y_var, data)

    plt.suptitle('Factors Influencing Confidence with Digital Fashion Technologies', fontsize=14, y=0.98)
    plt.figtext(0.01, 0.02, "NOTE: Ordinal data displayed with jitter for visibility. OLS lines shown for descriptive context only; ordinal/ordered logistic regression recommended for inference.",
               fontsize=7.5, color='#B5294E', style='italic')
    plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", fontsize=8)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18)  # Adjusted margins
    save_figure(fig, "confidence_factors_combined")
    plt.close()

def create_challenges_combined_figure(data):
    """Create a figure with four challenge-related regression plots.

    NOTE: Q11 (Challenges) is a 6-category nominal outcome. Jittered points displayed
    for visibility; OLS lines shown for descriptive context only."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.4)  # Increased spacing

    # Define variables for each plot
    plots = [
        ('Q12', 'Q11', 'Software Complexity', 'Challenges with 3D Design', 0, 0),
        ('Q13', 'Q11', 'Teacher Knowledge', 'Challenges with 3D Design', 0, 1),
        ('Q14_1', 'Q11', 'Software Availability', 'Challenges with 3D Design', 1, 0),
        ('Q14_2', 'Q11', 'Equipment Quality', 'Challenges with 3D Design', 1, 1)
    ]

    # Create each subplot
    for x_var, y_var, x_label, y_label, row, col in plots:
        ax = fig.add_subplot(gs[row, col])

        jitter_strength = 0.12
        x_jittered = data[x_var] + np.random.normal(0, jitter_strength, size=len(data))
        y_jittered = data[y_var] + np.random.normal(0, jitter_strength, size=len(data))
        ax.scatter(x_jittered, y_jittered, alpha=0.4, s=35,
                  color=categorical_palette[row * 2 + col], edgecolors='white', linewidths=0.3)

        # Add OLS line for descriptive context
        z = np.polyfit(data[x_var].dropna(), data[y_var].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[x_var].min(), data[x_var].max(), 100)
        ax.plot(x_line, p(x_line), color='red', lw=1.5, alpha=0.8, linestyle='--')

        # Calculate correlation
        corr, p_value = stats.pearsonr(data[x_var].dropna(), data[y_var].dropna())

        # Add significance stars
        sig_text = ""
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"

        # Add correlation annotation
        stats_text = f"r = {corr:.3f}{sig_text}\n(Pearson's r)"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Set axis labels with shorter wrapped text to avoid overlapping
        wrapped_x_label = textwrap.fill(x_label, width=20)  # Reduced width
        short_y_label = "3D Design Challenges"  # Shortened label
        wrapped_y_label = textwrap.fill(short_y_label, width=15)
        ax.set_xlabel(wrapped_x_label, fontsize=10, labelpad=10)
        ax.set_ylabel(wrapped_y_label, fontsize=10, labelpad=10)

        # Apply consistent axis labeling
        apply_axis_labels(ax, x_var, y_var, data)

    plt.suptitle('Factors Influencing Challenges with 3D Virtual Fashion Design', fontsize=14, y=0.98)
    plt.figtext(0.01, 0.02, "NOTE: Ordinal data displayed with jitter for visibility. OLS lines shown for descriptive context only; ordinal/multinomial regression recommended for inference.",
               fontsize=7.5, color='#B5294E', style='italic')
    plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", fontsize=8)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18)  # Adjusted margins
    save_figure(fig, "challenges_factors_combined")
    plt.close()

def create_investment_combined_figure(data):
    """Create a figure with four investment-related regression plots.

    NOTE: Q15 (Willingness to Invest) is a 5-point ordinal Likert variable. Q10 and Q11
    are also ordinal. Jittered points displayed for visibility; OLS lines shown for
    descriptive context only."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.4)  # Increased spacing

    # Define variables for each plot - removed Q14_4 which was causing the error
    plots = [
        ('Q16', 'Q15', 'Training Payment Willingness', 'Willingness to Invest in Software', 0, 0),
        ('Q17', 'Q15', 'Perceived Importance', 'Willingness to Invest in Software', 0, 1),
        ('Q10', 'Q15', 'Confidence with Digital Fashion Technologies', 'Willingness to Invest in Software', 1, 0),
        ('Q11', 'Q15', 'Challenges with 3D Design', 'Willingness to Invest in Software', 1, 1)
    ]

    # Create each subplot
    for x_var, y_var, x_label, y_label, row, col in plots:
        ax = fig.add_subplot(gs[row, col])

        jitter_strength = 0.12
        x_jittered = data[x_var] + np.random.normal(0, jitter_strength, size=len(data))
        y_jittered = data[y_var] + np.random.normal(0, jitter_strength, size=len(data))
        ax.scatter(x_jittered, y_jittered, alpha=0.4, s=35,
                  color=categorical_palette[row * 2 + col], edgecolors='white', linewidths=0.3)

        # Add OLS line for descriptive context
        z = np.polyfit(data[x_var].dropna(), data[y_var].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[x_var].min(), data[x_var].max(), 100)
        ax.plot(x_line, p(x_line), color='red', lw=1.5, alpha=0.8, linestyle='--')

        # Calculate correlation — Spearman appropriate for ordinal/Likert variables
        corr, p_value = stats.spearmanr(data[x_var].dropna(), data[y_var].dropna())

        # Add significance stars
        sig_text = ""
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"

        # Add correlation annotation
        stats_text = f"ρ = {corr:.3f}\n(Spearman's ρ) {sig_text}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Set axis labels with wrapped text
        wrapped_x_label = textwrap.fill(x_label, width=20)
        wrapped_y_label = textwrap.fill(y_label, width=20)
        ax.set_xlabel(wrapped_x_label, fontsize=10, labelpad=10)
        ax.set_ylabel(wrapped_y_label, fontsize=10, labelpad=10)

        # Apply consistent axis labeling
        apply_axis_labels(ax, x_var, y_var, data)

    plt.suptitle('Factors Influencing Willingness to Invest in Software', fontsize=14, y=0.98)
    plt.figtext(0.01, 0.02, "NOTE: Ordinal data displayed with jitter for visibility. OLS lines shown for descriptive context only; ordinal/ordered logistic regression recommended for inference.",
               fontsize=7.5, color='#B5294E', style='italic')
    plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", fontsize=8)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18)
    save_figure(fig, "investment_factors_combined")
    plt.close()

# Create pathway models for visualization
pathway_results = {}

print("\nRunning regression models for pathways...")
for pathway_id, pathway_info in pathways.items():
    # Extract variables
    dependent_var = pathway_info['dependent']
    independent_vars = pathway_info['independent']

    print(f"\nAnalyzing {pathway_info['title']}...")

    # Prepare data
    X = data_processed[independent_vars]
    y = data_processed[dependent_var]

    # Add constant
    X_with_const = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X_with_const).fit()

    # Store results
    pathway_results[str(pathway_info['title'])] = {
        'model': model,
        'dependent_var': dependent_var,
        'independent_vars': independent_vars,
    }

# Generate visualizations
print("\nGenerating visualizations...")

# Basic distributions for each dependent variable
for pathway_id, pathway_info in pathways.items():
    dep_var = pathway_info['dependent']
    create_distribution_plot(
        data_processed,
        dep_var,
        f"Distribution of {pathway_info['dependent_label']}",
        pathway_info['dependent_label']
    )

# Generate correlation heatmaps for each pathway
for pathway_id, pathway_info in pathways.items():
    variables = [str(pathway_info['dependent'])] + list(pathway_info['independent'])
    create_correlation_heatmap(
        data_processed,
        variables,
        f"Correlation Heatmap: {pathway_info['title']}",
        f"_{pathway_id.lower().replace(' ', '_')}"
    )

# Generate pathway interaction plots
create_pathway_interaction_plots(data_processed)

# Generate regression plots for each pathway
for pathway_id, pathway_info in pathways.items():
    dep_var = str(pathway_info['dependent'])
    dep_label = str(pathway_info['dependent_label'])

    # Create individual regression plots
    for ind_var in pathway_info['independent']:
        ind_label = str(pathway_info['independent_labels'][ind_var])
        create_regression_plot(
            data_processed,
            ind_var,
            dep_var,
            ind_label,
            dep_label,
            f"Relationship between {ind_label} and {dep_label}",
            f"_{pathway_id.lower().replace(' ', '_')}"
        )

    # Create multi-regression panel
    create_multi_regression_plot(
        data_processed,
        list(pathway_info['independent']),
        dep_var,
        f"Factors Influencing {dep_label}",
        f"_{pathway_id.lower().replace(' ', '_')}"
    )

    # Create correlation network for pathway variables
    all_vars = [dep_var] + list(pathway_info['independent'])
    create_correlation_network(
        data_processed,
        all_vars,
        f"Correlation Network: {pathway_info['title']}",
        f"_{pathway_id.lower().replace(' ', '_')}"
    )

# Create coefficient (hypothesis testing) plots for each pathway model
for pathway_name, results in pathway_results.items():
    create_coefficient_plot(
        results['model'],
        f"Hypothesis Testing: {pathway_name}",
        "hypothesis_testing_" + pathway_name.lower().replace(' ', '_').replace(':', '')
    )

    create_residual_plots(
        results['model'],
        pathway_name,
        pathway_name.lower().replace(' ', '_').replace(':', '')
    )

# Create combined visualization of all pathways
all_vars = []
for pathway_info in pathways.values():
    all_vars.append(pathway_info['dependent'])
    all_vars.extend(pathway_info['independent'])

# Remove duplicates while preserving order
all_vars = list(dict.fromkeys(all_vars))

# Create correlation heatmap for all variables
create_correlation_heatmap(
    data_processed,
    all_vars,
    "Correlation Heatmap: All Pathway Variables",
    "_all_pathways"
)

# Create correlation network for all variables
create_correlation_network(
    data_processed,
    all_vars,
    "Correlation Network: All Pathway Variables",
    "_all_pathways"
)

# Create combined figures with regression plots
print("\nGenerating improved combined visualization figures...")

# Create combined figures with proper axis labels
create_confidence_combined_figure(data_processed)
create_challenges_combined_figure(data_processed)
create_investment_combined_figure(data_processed)

# The previous combined plots with improved axis labels
confidence_pairs = [
    ('Q7', 'Q10', 'Experience Level', 'Confidence with Digital Fashion Technologies'),
    ('Q8', 'Q10', 'Proficiency Level', 'Confidence with Digital Fashion Technologies'),
    ('Q9_1', 'Q10', 'YouTube Tutorial Usefulness', 'Confidence with Digital Fashion Technologies'),
    ('Q9_2', 'Q10', 'Online Resource Usefulness', 'Confidence with Digital Fashion Technologies')
]
create_combined_plot(data_processed, confidence_pairs,
                    'Factors Influencing Confidence with Digital Fashion Technologies',
                    'confidence_combined')

# Combined figure for Challenge factors with shortened labels
challenge_pairs = [
    ('Q12', 'Q11', 'Software Complexity', 'Challenges with 3D Design'),
    ('Q13', 'Q11', 'Teacher Knowledge', 'Challenges with 3D Design'),
    ('Q14_1', 'Q11', 'Technical Resources', 'Challenges with 3D Design'),
    ('Q14_2', 'Q11', 'Equipment Resources', 'Challenges with 3D Design')
]
create_combined_plot(data_processed, challenge_pairs,
                    'Factors Influencing Challenges with 3D Virtual Fashion Design',
                    'challenges_combined')

# Combined figure for Willingness to Invest factors
invest_pairs = [
    ('Q16', 'Q15', 'Training Payment Willingness', 'Willingness to Invest in Software'),
    ('Q17', 'Q15', 'Perceived Importance', 'Willingness to Invest in Software'),
    ('Q14_1', 'Q15', 'Technical Resources', 'Willingness to Invest in Software'),
    ('Q14_3', 'Q15', 'Support Resources', 'Willingness to Invest in Software')
]
create_combined_plot(data_processed, invest_pairs,
                    'Factors Influencing Willingness to Invest in Software',
                    'invest_combined')

print("\nAll visualizations completed and saved to the figures directory.")

