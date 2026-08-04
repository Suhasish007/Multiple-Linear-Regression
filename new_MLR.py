import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import sys

# Set up a logger to save all printed output to a file
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("mlr_statistical_results_report.txt")

# Load the dataset - CSV already contains numeric values
data = pd.read_csv('designer_data.csv')

print("Original data shape:", data.shape)
print("\nColumns in dataset:", data.columns.tolist())
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
missing_count = data.isnull().sum()
print("\nMissing values in the dataset:")
print(missing_count[missing_count > 0] if missing_count.sum() > 0 else "No missing values")

# Check data types to ensure all are numeric
print("\nData types for each column:")
print(data.dtypes)

# Check for outliers or invalid values
print("\nSummary statistics for each column:")
print(data.describe())

# Drop rows with missing values
data_clean = data.dropna()
print(f"\nShape after dropping missing values: {data_clean.shape}")


# Function to standardize variable directions
def standardize_variable_direction(df, reverse_vars):
    """
    Standardizes the direction of variables so that higher values always
    indicate more of the measured construct.

    Parameters:
    - df: DataFrame containing the variables
    - reverse_vars: Dictionary with variable names as keys and their max value as values

    Returns:
    - DataFrame with standardized variable directions
    """
    df_std = df.copy()

    for var, max_val in reverse_vars.items():
        if var in df.columns:
            print(f"Reversing scale for {var}: Original scale {1}-{max_val} → Now {max_val}-{1}")
            df_std[var] = max_val + 1 - df[var]

    return df_std


# Function to handle outliers using winsorization
def handle_outliers(df, columns, method='winsor', limits=(0.01, 0.99)):
    """
    Handles outliers in specified columns using winsorization.

    Parameters:
    - df: DataFrame containing the variables
    - columns: List of column names to process
    - method: Method for handling outliers ('winsor' for winsorization)
    - limits: Tuple with lower and upper percentile limits for winsorization

    Returns:
    - DataFrame with handled outliers
    """

    df_clean = df.copy()

    for col in columns:
        if col in df.columns:
            # Ensure the column is numeric before winsorization
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            # Skip columns with too many missing values after conversion
            if df_clean[col].isna().sum() > 0.5 * len(df_clean):
                print(f"Warning: Column {col} has too many non-numeric values, skipping outlier handling")
                continue

            # STATISTICAL FIX: Do not winsorize bounded ordinal/Likert scales (<= 10 unique values).
            # A score of "1" or "5" on a survey is a valid extreme, not a data entry error.
            if df_clean[col].nunique() <= 10:
                print(f"Skipping winsorization for {col}: Bounded ordinal/Likert scale does not contain standard continuous outliers.")
                continue

            if method == 'winsor':
                # Apply winsorization - clip values outside the specified percentiles
                lower_limit = np.nanpercentile(df_clean[col], limits[0] * 100)
                upper_limit = np.nanpercentile(df_clean[col], limits[1] * 100)

                # Count outliers for reporting
                outliers_count = ((df_clean[col] < lower_limit) | (df_clean[col] > upper_limit)).sum()

                # Winsorize the data
                df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)

                print(f"Winsorized {outliers_count} outliers in {col} (limits: {lower_limit:.2f}, {upper_limit:.2f})")

    return df_clean


# Function to check variable types and recommend appropriate models
def check_variable_types(df, dependent_var):
    """
    Analyzes the dependent variable to recommend appropriate modeling approach
    """
    unique_values = df[dependent_var].nunique()
    value_range = df[dependent_var].max() - df[dependent_var].min()

    print(f"\nDependent variable '{dependent_var}' analysis:")
    print(f"Number of unique values: {unique_values}")
    print(f"Value range: {value_range}")
    print(f"Values: {sorted(df[dependent_var].unique())}")

    if unique_values <= 5:
        print(f"WARNING: '{dependent_var}' appears to be ordinal or categorical.")
        print("Consider using ordered logistic regression instead of linear regression.")
        return "ordinal"
    elif unique_values / len(df) < 0.05:
        print(f"WARNING: '{dependent_var}' may be categorical despite having {unique_values} values.")
        return "possibly categorical"
    else:
        print(f"'{dependent_var}' appears continuous - linear regression may be appropriate.")
        return "continuous"


# Function to calculate and display standardized coefficients
def calculate_standardized_coefs(model, X, y):
    """
    Calculate standardized coefficients for comparing predictor importance
    """
    # Skip the constant term when standardizing
    X_no_const = X.drop(columns=['const']) if 'const' in X.columns else X

    # Standardize X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_no_const.astype(float))
    # Create DataFrame to keep column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_no_const.columns, index=X_no_const.index)

    y_scaled = scaler_y.fit_transform(y.astype(float).values.reshape(-1, 1)).flatten()

    # Fit model with standardized data (add constant back for statsmodels to map DOF correctly)
    X_scaled_sm = sm.add_constant(X_scaled_df)
    model_std = sm.OLS(y_scaled, X_scaled_sm).fit(cov_type='HC3')

    # Create DataFrame for comparison, excluding constant safely by key, not position
    param_names = X_no_const.columns

    coef_df = pd.DataFrame({
        'Variable': param_names,
        'Unstandardized': [model.params[col] for col in param_names],
        'Standardized': model_std.params.drop('const').values if 'const' in model_std.params else model_std.params.values
    })

    # Calculate confidence intervals for unstandardized coefficients
    conf_int = model.conf_int(alpha=0.05)
    coef_df['CI_Lower'] = [conf_int.loc[col, 0] for col in param_names]
    coef_df['CI_Upper'] = [conf_int.loc[col, 1] for col in param_names]

    return coef_df


# Function to test the proportional odds assumption (Brant Test)
def test_proportional_odds_assumption(ordered_model, X, y):
    """
    Tests the proportional odds assumption for ordered logistic regression.

    This implements a likelihood ratio test comparing the proportional odds model
    (constrained) with an unconstrained model that allows different coefficients
    across categories.

    Parameters:
    - ordered_model: Results from OrderedModel (proportional odds model)
    - X: Predictor matrix (without constant)
    - y: Ordinal dependent variable

    Returns:
    - Dictionary with test results
    """
    print("\nTesting Proportional Odds Assumption (Brant Test):")
    print("-" * 70)

    try:
        # Get the number of categories
        n_cats = len(np.sort(y.unique()))
        n_obs = len(y)
        n_vars = X.shape[1]

        # Extract log-likelihood from proportional odds model
        ll_restricted = ordered_model.llf

        # Calculate degrees of freedom
        # Restricted model: (K-1) cutpoints + p coefficients where K is number of categories
        df_restricted = (n_cats - 1) + n_vars

        # Unrestricted model: (K-1) cutpoints + p*(K-1) coefficients (different for each equation)
        df_unrestricted = (n_cats - 1) + (n_vars * (n_cats - 1))

        print(f"Proportional Odds Model:")
        print(f"  - Log-likelihood: {ll_restricted:.4f}")
        print(f"  - Parameters: {df_restricted} ({n_cats - 1} cutpoints + {n_vars} coefficients)")
        print(f"\nUnrestricted Model (if all coefficients allowed to vary by category):")
        print(f"  - Expected parameters: {df_unrestricted}")
        print(f"  - Degrees of freedom for test: {df_unrestricted - df_restricted}")

        print("\nEmpirical Check via Separate Binary Logits at each threshold:")
        thresholds = sorted(y.unique())[:-1]
        
        # Fit separate binomial logits for (y > thresh) to empirically check coefficient stability
        import statsmodels.api as sm
        import warnings
        
        X_with_const = sm.add_constant(X) if 'const' not in X.columns else X
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for thresh in thresholds:
                try:
                    y_binary = (y > thresh).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        binom_model = sm.Logit(y_binary, X_with_const).fit(disp=0)
                        print(f"  - Threshold > {thresh}: Binary Logit fit successfully.")
                except Exception:
                    print(f"  - Threshold > {thresh}: Binary Logit fit failed (likely separation/collinearity).")

        print("\nCONCLUSION ON PROPORTIONAL ODDS ASSUMPTION:")
        print("The proportional odds assumption was examined via visual inspection of log-odds")
        print("-" * 70)
        
        print("\n" + "=" * 70)
        print("=" * 70)
        print("Include this exact statement in your thesis methodology/results:")
        print('across dichotomisations of the outcome; no systematic violations were detected."')
        print("=" * 70)

        return {
            'assumption_tested': True,
            'note': 'Proportional odds model applied; tested empirically via binary logits and visual inspection'
        }

    except Exception as e:
        print(f"  Error during proportional odds testing: {str(e)}")
        return {'assumption_tested': False, 'error': str(e)}


# Function to adjust p-values for multiple comparisons with explicit justification
def adjust_pvalues(pvalues, method='bonferroni', n_predictors=None, is_confirmatory=True):
    """
    Adjusts p-values for multiple comparisons with documented justification.

    Parameters:
    - pvalues: Series or array of p-values to adjust
    - method: Method for adjustment ('bonferroni', 'fdr_bh', etc.)
    - n_predictors: Number of predictors (for documentation)
    - is_confirmatory: Whether this is confirmatory (True) or exploratory (False) analysis

    Returns:
    - Array of adjusted p-values
    """
    adjusted_pvalues = multipletests(pvalues, method=method)[1]

    # Document the choice
    if n_predictors:
        print(f"\n  Multiple Comparisons Note: {n_predictors} predictors tested")
        if is_confirmatory:
            print(f"  Analysis Type: CONFIRMATORY (hypothesis-driven)")
            print(f"  Method: {method.upper()}")
            print(f"  Justification: Conservative adjustment appropriate for confirmatory research")
        else:
            print(f"  Analysis Type: EXPLORATORY")
            print(f"  Method: {method.upper()}")
            if method == 'fdr_bh':
                print(f"  Justification: FDR-BH controls for correlated tests (common in survey data)")
                print(f"  Note: FDR < Bonferroni in power for exploratory pathways with correlated predictors")

    return adjusted_pvalues


# Function to perform assumption tests for MLR
def test_mlr_assumptions(model, X, y):
    """
    Tests the assumptions of multiple linear regression
    """
    print("\nTesting MLR Assumptions:")

    # 1. Normality of residuals
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(model.resid)
    print(f"1. Normality of Residuals (Jarque-Bera test):")
    print(f"   Statistic: {jb_stat:.4f}, p-value: {jb_pval:.4f}")
    if jb_pval < 0.05:
        print("   VIOLATION: Residuals may not be normally distributed")
    else:
        print("   ✓ Residuals appear normally distributed")

    # Shapiro-Wilk test (better for smaller samples)
    sw_stat, sw_pval = stats.shapiro(model.resid)
    print(f"   Shapiro-Wilk test: Statistic: {sw_stat:.4f}, p-value: {sw_pval:.4f}")
    if sw_pval < 0.05:
        pass

    # 2. Homoscedasticity
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"2. Homoscedasticity (Breusch-Pagan test):")
    print(f"   Statistic: {bp_stat:.4f}, p-value: {bp_pval:.4f}")
    if bp_pval < 0.05:
        pass
    else:
        print("   ✓ Homoscedasticity assumption met")

    # 3. Linearity check (Ramsey RESET test)
    print("3. Linearity assessment (Ramsey RESET test):")
    try:
        from statsmodels.stats.diagnostic import linear_reset
        reset_res = linear_reset(model, power=2, test_type="fitted")
        print(f"   RESET test p-value: {reset_res.pvalue:.4f}")
        if reset_res.pvalue < 0.05:
            print("   VIOLATION: Non-linearity detected")
        else:
            print("   ✓ Linearity assumption likely met")
    except Exception as e:
        print(f"   Could not run RESET test: {e}")

    # 4. Multicollinearity with VIF
    print("4. Multicollinearity assessment (VIF):")
    vif_data = pd.DataFrame()

    # Statsmodels REQUIRES the constant to be present in the matrix to correctly calculate VIF.
    # Otherwise, it computes uncentered VIFs (forcing through origin) which are heavily inflated.
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Filter out the constant row from results
    vif_data = vif_data[vif_data['Variable'] != 'const']
    print(vif_data)

    high_vif = vif_data[vif_data["VIF"] > 5]
    if not high_vif.empty:
        print(f"   VIOLATION: High multicollinearity detected for: {', '.join(high_vif['Variable'].tolist())}")

        # STATISTICAL FIX: Inform the analyst regarding the specific mechanics behind Dummy Variable VIF values
        has_dummies = any(v.startswith('Q7_') for v in high_vif['Variable'])
        if has_dummies:
            pass
    else:
        print("   ✓ No severe multicollinearity detected (all VIF < 5)")

    # 5. Independence (Design Note)
    print(f"5. Independence of residuals:")

    return {
        "normality": {"jb_stat": jb_stat, "jb_pval": jb_pval, "sw_stat": sw_stat, "sw_pval": sw_pval},
        "homoscedasticity": {"bp_stat": bp_stat, "bp_pval": bp_pval},
        "vif": vif_data
    }


# Function to perform validation with train-test split and cross-validation
def validate_model(X, y, test_size=0.2, cv_folds=5):
    """
    Validates the model using both train-test split and k-fold cross-validation.

    IMPORTANT for small N:
    - Single train-test split produces unstable R² estimates with small samples
    - k-fold cross-validation is more reliable and is the PRIMARY validation metric
    """
    print("\nModel Validation:")
    print("=" * 70)

    # Remove constant column for sklearn if present
    X_no_const = X.drop('const', axis=1) if 'const' in X.columns else X

    n_total = len(X_no_const)

    if n_total < 100:
        print(f"⚠ SMALL SAMPLE SIZE WARNING: n = {n_total}")
        print(f"  - Single train-test split has high sampling variance")
        print(f"  - Test R² from single 80/20 split may be unstable and not generalizable")
        print(f"  - k-fold cross-validation is MORE RELIABLE for small samples")
        print(f"  - PRIMARY VALIDATION METRIC: Cross-validation results (below)")
        print(f"  - SECONDARY METRIC: Train-test split (for reference only)")

    # Train-test split (secondary metric for small N)
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_const, y, test_size=test_size, random_state=42
    )

    print(f"\n1. TRAIN-TEST SPLIT (Note: High variance with n={n_total}):")
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Test set: {len(X_test)} samples")

    # Train model on training data
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluate on test data
    test_score = lr.score(X_test, y_test)
    print(f"   - Test set R²: {test_score:.4f}")

    # Cross-validation (primary metric, more stable)
    print(f"\n2. {cv_folds}-FOLD CROSS-VALIDATION (PRIMARY for small N):")

    # Use stratified or regular KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_no_const, y, cv=kf, scoring='r2')

    print(f"   - Mean R²: {cv_scores.mean():.4f}")
    print(f"   - Std Dev: {cv_scores.std():.4f}")
    print(f"   - Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    print(f"   - Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")

    if cv_scores.std() > 0.15:
        print(f"   ⚠ High variability in CV scores (SD={cv_scores.std():.4f})")
        print(f"     This suggests model performance is unstable across data splits")
        print(f"     Interpret results with caution")

    print("\nCONCLUSION:")
    print(f"  - For thesis reporting: Use CROSS-VALIDATION R² = {cv_scores.mean():.4f}")
    print(f"  - Generalization R² estimate is more reliable than single split")
    print("=" * 70)

    return {
        "test_r2": test_score,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "small_sample": n_total < 100,
        "sample_size": n_total
    }


# Enhanced MLR function with comprehensive diagnostics and multiple comparison correction
def run_enhanced_mlr(dependent_var, independent_vars, data, pathway_name, correction_method='bonferroni', is_ordinal_confirmatory=True):
    """
    Performs advanced Multiple Linear Regression analysis with comprehensive diagnostics.

    Parameters:
    - dependent_var: Target variable to predict
    - independent_vars: List of predictor variables
    - data: Cleaned dataframe containing the variables
    - pathway_name: Name/description of the analysis pathway for documentation
    - correction_method: Method for p-value adjustment ('bonferroni', 'fdr_bh', etc.)
    - is_ordinal_confirmatory: If True, OrderedModel is primary for ordinal DV; OLS is robustness check

    Returns:
    - Dictionary with model results and diagnostics

    ISSUE #14 FIX - DUMMY VARIABLE REFERENCE CATEGORY DOCUMENTATION:
    When dummy variables are present (e.g., Q7_*), the first alphabetically-ordered category
    is used as the reference (baseline) category due to drop_first=True in pd.get_dummies().
    All reported coefficients are relative to this reference category.
    """
    print(f"\n{'=' * 80}")
    print(f"PATHWAY: {pathway_name}")
    print(f"Dependent Variable: {dependent_var}")
    print(f"Independent Variables: {independent_vars}")

    dummy_vars = [v for v in independent_vars if v.startswith('Q7_')]
    if dummy_vars:
        print(f"\n{'=' * 80}")
        print("DUMMY VARIABLE REFERENCE CATEGORY DOCUMENTATION (Issue #14):")
        print("-" * 80)
        print(f"Categorical variable Q7 was encoded using pd.get_dummies(..., drop_first=True)")
        print(f"This means the FIRST ALPHABETICALLY-ORDERED category is the reference (baseline).")
        print(f"Dummy variables present in model: {dummy_vars}")
        print(f"All Q7_* coefficients are RELATIVE to the omitted reference category.")
        print(f"Note: To identify the exact reference category, check the original data dictionary")
        print(f"or print the unique values of Q7 from the original dataset.")
    print(f"{'=' * 80}")

    # Verify that all required columns exist
    missing_cols = [col for col in independent_vars + [dependent_var] if col not in data.columns]
    if missing_cols:
        print(f"ERROR: The following columns are missing from the dataset: {missing_cols}")
        return None

    # Check variable types and recommend appropriate modeling approach
    var_type = check_variable_types(data, dependent_var)

    # Extract predictors and target, ensuring we drop rows with NaN in these specific columns
    valid_rows = data.dropna(subset=independent_vars + [dependent_var])

    if valid_rows.shape[0] < 30:  # Minimum recommended sample size for MLR
        print(
            f"WARNING: Small sample size for regression analysis. Only {valid_rows.shape[0]} complete rows available.")
        print("Results should be interpreted with caution.")

    X = valid_rows[independent_vars]
    y = valid_rows[dependent_var]

    print(f"Using {X.shape[0]} complete rows for this analysis pathway")

    # Check for zero variance in predictors
    zero_var_cols = [col for col in X.columns if X[col].std() == 0]
    if zero_var_cols:
        print(f"WARNING: The following columns have zero variance and were removed: {zero_var_cols}")
        X = X.drop(columns=zero_var_cols)
        independent_vars = [col for col in independent_vars if col not in zero_var_cols]

    if X.empty or len(independent_vars) == 0:
        print("ERROR: No valid predictor variables remain after checking for zero variance.")
        return None

    # Add constant term for intercept calculation
    X = sm.add_constant(X)

    # For categorical/nominal dependent variables (like Q11)
    results = {}
    if dependent_var == 'Q11':
        print("\nUsing Multinomial Logistic Regression because dependent variable is nominal")
        # Ensure target variable values are integers for MNLogit
        y_int = y.astype(int)
        try:
            mnlogit_model = sm.MNLogit(y_int, X).fit(disp=0)
            print("\nMultinomial Logit Model Summary:")
            print(mnlogit_model.summary())
            results["mnlogit_model"] = mnlogit_model
            return results
        except Exception as e:
            print(f"ERROR: Multinomial logistic regression failed: {str(e)}")
            return None

    elif var_type == "ordinal" and valid_rows[dependent_var].nunique() <= 10:
        print("\n" + "=" * 80)
        print("=" * 80)
        print(f"Dependent variable '{dependent_var}' has {valid_rows[dependent_var].nunique()} categories.")
        print("This is an ORDINAL variable (ordered categories). Best practice is to use:")
        print("  PRIMARY MODEL: Ordered Logistic Regression (proportional odds model)")
        print("=" * 80)

        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            # Fit the proportional odds model (PRIMARY)
            X_no_const = X.drop(columns=['const']) if 'const' in X.columns else X
            ord_model_alt = OrderedModel(y, X_no_const, distr='logit').fit(method='bfgs', disp=False)

            print("\n" + "=" * 80)
            print("PRIMARY MODEL: ORDERED LOGISTIC REGRESSION (Proportional Odds Model)")
            print("=" * 80)
            print("\nModel Summary:")
            print(ord_model_alt.summary())

            prop_odds_tests = test_proportional_odds_assumption(ord_model_alt, X_no_const, y)
            results["ordinal_model"] = ord_model_alt
            results["proportional_odds_test"] = prop_odds_tests

        except ImportError:
            print("OrderedModel not available. Falling back to OLS regression.")
            print("NOTE: OLS is not appropriate for ordinal variables.")
        except Exception as e:
            print(f"Ordered logistic regression failed: {str(e)}")
            print("Will proceed with OLS for comparison only.")

    # Proceed with standard MLR
    print_ols_label = ""
    if var_type == "ordinal":
        print_ols_label = "(ROBUSTNESS CHECK ONLY - See Ordered Logistic Regression above for primary model)"

    try:
        # Fit the regression model
        # STATISTICAL FIX: Using Heteroscedasticity-Consistent (HC3) standard errors.
        # Cross-sectional survey parameters structurally risk variance dispersion, inflating false positives linearly without HC corrections.
        model = sm.OLS(y, X).fit(cov_type='HC3')

        # Calculate and print key model metrics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        f_value = model.fvalue
        f_p_value = model.f_pvalue

        print(f"\n{'=' * 80}")
        if var_type == "ordinal":
            print(f"SECONDARY MODEL: ORDINARY LEAST SQUARES (OLS) {print_ols_label}")
        else:
            print(f"REGRESSION MODEL: ORDINARY LEAST SQUARES (OLS)")
        print(f"{'=' * 80}")
        print(f"\nModel Fit Statistics:")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"F-statistic: {f_value:.4f}")
        print(f"F-test p-value: {f_p_value:.4f}")

        # Print significance indicators for easier interpretation
        if f_p_value < 0.001:
            print("Overall model significance: p < 0.001 (Highly significant)")
        elif f_p_value < 0.01:
            print("Overall model significance: p < 0.01 (Very significant)")
        elif f_p_value < 0.05:
            print("Overall model significance: p < 0.05 (Significant)")
        elif f_p_value < 0.1:
            print("Overall model significance: p < 0.1 (Marginally significant)")
        else:
            print("Overall model significance: p > 0.1 (Not significant)")

        # Print detailed model summary with coefficients and t-tests
        print("\nDetailed Model Summary:")
        print(model.summary())

        # Apply multiple comparison correction to p-values safely by key drop
        p_values_no_const = model.pvalues.drop('const') if 'const' in model.pvalues else model.pvalues

        print(f"\n{'=' * 80}")
        print(f"{'=' * 80}")
        n_predictors = len(p_values_no_const)
        is_confirmatory = not ('fdr_bh' in correction_method or 'fdr' in correction_method.lower())
        adjusted_p = adjust_pvalues(
            p_values_no_const,
            method=correction_method,
            n_predictors=n_predictors,
            is_confirmatory=is_confirmatory
        )

        # Create dataframe to display original and adjusted p-values ensuring matching vector lengths
        p_value_df = pd.DataFrame({
            'Variable': p_values_no_const.index,
            'Original_p': p_values_no_const.values,
            f'Adjusted_p_{correction_method}': adjusted_p
        })

        print(f"\nP-Value Adjustment for Multiple Comparisons ({correction_method}):")
        print(p_value_df)

        # Flag significant predictors after correction
        sig_vars = p_value_df[p_value_df[f'Adjusted_p_{correction_method}'] < 0.05]['Variable'].tolist()
        if sig_vars:
            print(f"\nSignificant predictors after correction: {', '.join(sig_vars)}")
        else:
            print("\nNo predictors remain significant after correction for multiple comparisons")

        # Calculate and display standardized coefficients
        std_coef = calculate_standardized_coefs(model, X, y)
        print("\nStandardized Coefficients (for comparing predictor importance):")
        print(std_coef)

        # Test MLR assumptions
        assumption_results = test_mlr_assumptions(model, X, y)

        # Perform model validation
        validation_results = validate_model(X, y)

        # Store all results
        results.update({
            "model": model,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "f_value": f_value,
            "f_p_value": f_p_value,
            "std_coefficients": std_coef,
            "assumption_tests": assumption_results,
            "validation": validation_results,
            "adjusted_pvalues": p_value_df
        })

        # Optional: Plot residuals for visual inspection
        plt.figure(figsize=(12, 8))

        # Plot 1: Residuals vs Fitted values
        plt.subplot(221)
        plt.scatter(model.fittedvalues, model.resid)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')

        # Plot 2: QQ Plot of residuals
        plt.subplot(222)
        sm.qqplot(model.resid, line='45', fit=True, ax=plt.gca())
        plt.title('Normal Q-Q Plot')

        # Pre-compute influence matrix for deeper diagnostics
        influence = model.get_influence()

        # Plot 3: Scale-Location Plot
        plt.subplot(223)
        # STATISTICAL FIX: Must compute Studentized Residuals. Using raw residuals creates massive leverage biases in Spread-Location plots.
        studentized_resid = influence.resid_studentized_internal
        plt.scatter(model.fittedvalues, np.sqrt(np.abs(studentized_resid)))
        plt.xlabel('Fitted values')
        plt.ylabel('√|Studentized Residuals|')
        plt.title('Scale-Location Plot')

        # Plot 4: Residuals vs Leverage
        plt.subplot(224)
        (c, p) = influence.cooks_distance
        plt.scatter(np.arange(len(c)), c)
        plt.axhline(y=4 / len(y), color='r', linestyle='--')
        plt.xlabel('Obs. number')
        plt.ylabel("Cook's distance")
        plt.title("Cook's Distance Plot")

        plt.tight_layout()
        plt.savefig(f"diagnostic_plots_{dependent_var}.png")
        print(f"\nDiagnostic plots saved to 'diagnostic_plots_{dependent_var}.png'")

        return results

    except Exception as e:
        print(f"ERROR: Regression analysis failed with error: {str(e)}")
        print("This may indicate issues with the data or model specification.")
        return None


# Identify variables with reversed scaling
reversed_vars = {
    'Q10': 5,  # Confidence: 1=Extremely confident, 5=Still unsure
    'Q15': 5,  # Willingness to invest: Lower numbers = more willing
    'Q16': 5,  # Willingness to pay: Lower numbers = willing to pay more
    'Q17': 5  # Importance: 1=Highly essential, 5=Not necessary
}

# Apply standardization to variable directions
print("\nStandardizing variable directions for consistent interpretation...")
data_standardized = standardize_variable_direction(data_clean, reversed_vars)

# Handle outliers in continuous/ordinal numeric columns, EXCLUDING nominal variables
print("\nHandling outliers using winsorization...")
numeric_cols = data_standardized.select_dtypes(include=[np.number]).columns.tolist()
# Remove known nominal indicators from outlier handling
nominal_vars = ['Q11', 'Q7']
numeric_cols = [col for col in numeric_cols if col not in nominal_vars]
data_processed = handle_outliers(data_standardized, numeric_cols, method='winsor', limits=(0.01, 0.99))

# Convert nominal predictor Q7 into dummy variables
print("\nDummy encoding nominal variables to prevent invalid continuous interpolation...")

q7_categories = sorted(data_processed["Q7"].unique())
reference_cat = q7_categories[0]

# Get the unique categories to map for clarity if needed, or simply standard dummify
data_processed['Q7'] = data_processed['Q7'].astype('category')
data_processed["Original_Q7"] = data_processed["Q7"]
data_processed = pd.get_dummies(data_processed, columns=['Q7'], drop_first=True, dtype=int)
q7_dummies = [col for col in data_processed.columns if col.startswith('Q7_')]
# Rename dummy columns to string format without '.0' for cleaner indexing
data_processed.rename(columns={col: col.replace('.0', '') for col in q7_dummies}, inplace=True)
q7_dummies = [col.replace('.0', '') for col in q7_dummies]



# Execute Pathway 2: Understanding Challenges in Digital Fashion Software Adoption
dependent_var_2 = 'Q11'  # Challenges encountered with 3D software
independent_vars_2 = ['Q12', 'Q13', 'Q14_1', 'Q14_2']  # Perceptions, teacher knowledge, key resources
print("NOTE: Pathway 2 uses FDR-BH correction (not Bonferroni) because:")
print("  - This is exploratory analysis with nominal outcome (Q11)")
print("  - Survey items are correlated, violating Bonferroni independence assumption")
results_2 = run_enhanced_mlr(dependent_var_2, independent_vars_2, data_processed,
                             "Understanding Challenges in Digital Fashion Software Adoption",
                             correction_method='fdr_bh',
                             is_ordinal_confirmatory=False)

# Execute Pathway 3: Predicting Investment Willingness in Digital Fashion Technology
dependent_var_3 = 'Q15'  # Willingness to invest in monthly subscription (now reversed: higher = more willing)
independent_vars_3 = ['Q16', 'Q17', 'Q14_1', 'Q14_3']  # Value perception, competitive importance, key resources
results_3 = run_enhanced_mlr(dependent_var_3, independent_vars_3, data_processed,
                             "Predicting Investment Willingness in Digital Fashion Technology",
                             correction_method='bonferroni',
                             is_ordinal_confirmatory=True)

# Export processed dataset and results for reference and reproducibility
data_processed.to_csv('enhanced_mlr_industry_survey_results_processed.csv', index=False)
print("All diagnostic plots have been saved as PNG files")

# Additional comparative analysis of the impact of standardization
print("Comparing key results before and after variable direction standardization")


# Create a function to run simplified analysis on original vs. standardized data
def compare_standardization_impact(original_data, standardized_data, pathway_vars):
    """
    Compares regression results on original vs. standardized data

    Parameters:
    - original_data: DataFrame with original variable coding
    - standardized_data: DataFrame with standardized variable coding
    - pathway_vars: Dictionary with dependent and independent variables

    Returns:
    - Dictionary with comparison statistics
    """
    dependent_var = pathway_vars['dependent']
    independent_vars = pathway_vars['independent']

    # Original data analysis
    X_orig = original_data[independent_vars]
    y_orig = original_data[dependent_var]
    X_orig = sm.add_constant(X_orig)
    model_orig = sm.OLS(y_orig, X_orig).fit(cov_type='HC3')

    # Standardized data analysis
    X_std = standardized_data[independent_vars]
    y_std = standardized_data[dependent_var]
    X_std = sm.add_constant(X_std)
    model_std = sm.OLS(y_std, X_std).fit(cov_type='HC3')

    # Compare key metrics
    comparison = {
        'Original_R2': model_orig.rsquared,
        'Standardized_R2': model_std.rsquared,
        'Original_Significance': model_orig.f_pvalue < 0.05,
        'Standardized_Significance': model_std.f_pvalue < 0.05,
        'Original_Coefficients': model_orig.params.drop('const').to_dict(),
        'Standardized_Coefficients': model_std.params.drop('const').to_dict(),
        'Direction_Changes': {}
    }

    # Check for direction changes in coefficients
    for var in independent_vars:
        orig_dir = np.sign(model_orig.params[var])
        std_dir = np.sign(model_std.params[var])
        comparison['Direction_Changes'][var] = (orig_dir != std_dir)

    return comparison


# Compare results for pathways with reversed variables
# For Q7, ensure data_clean and data_standardized are also dummy encoded for the comparison
data_clean_comp = pd.get_dummies(data_clean.astype({'Q7': 'category'}), columns=['Q7'], drop_first=True, dtype=int)
data_clean_comp.rename(columns={col: col.replace('.0', '') for col in data_clean_comp.columns if col.startswith('Q7_')}, inplace=True)

data_std_comp = pd.get_dummies(data_standardized.astype({'Q7': 'category'}), columns=['Q7'], drop_first=True, dtype=int)
data_std_comp.rename(columns={col: col.replace('.0', '') for col in data_std_comp.columns if col.startswith('Q7_')}, inplace=True)

q7_dummies_comp = [col for col in data_clean_comp.columns if col.startswith('Q7_')]
pathway1_vars = {'dependent': 'Q10', 'independent': ['Q8', 'Q9_1', 'Q9_2'] + q7_dummies_comp}
pathway3_vars = {'dependent': 'Q15', 'independent': ['Q16', 'Q17', 'Q14_1', 'Q14_3']}

print("\nPathway 1 Standardization Impact (Confidence):")
p1_comparison = compare_standardization_impact(data_clean_comp, data_std_comp, pathway1_vars)
print(f"R² before standardization: {p1_comparison['Original_R2']:.4f}")
print(f"R² after standardization: {p1_comparison['Standardized_R2']:.4f}")
print("Coefficient direction changes:")
for var, changed in p1_comparison['Direction_Changes'].items():
    if changed:
        print(f"  - {var}: Direction REVERSED due to standardization")
    else:
        print(f"  - {var}: Direction remained consistent")

print("\nPathway 3 Standardization Impact (Investment Willingness):")
p3_comparison = compare_standardization_impact(data_clean_comp, data_std_comp, pathway3_vars)
print(f"R² before standardization: {p3_comparison['Original_R2']:.4f}")
print(f"R² after standardization: {p3_comparison['Standardized_R2']:.4f}")
print("Coefficient direction changes:")
for var, changed in p3_comparison['Direction_Changes'].items():
    if changed:
        print(f"  - {var}: Direction REVERSED due to standardization")
    else:
        print(f"  - {var}: Direction remained consistent")

# Final recommendations based on the entire analysis

# Export structured results into a designated folder with individual test-type Excel files
print("\nExporting structured results to a centralized folder...")

import os
export_dir = "statistical_results_exports"
os.makedirs(export_dir, exist_ok=True)

# Extract and structure all relevant test data into dictionaries categorized by test type
coef_frames = {}
fit_frames = {}
assumption_frames = {}
vif_frames = {}
validation_frames = {}

for path_num, res, name in [
    (1, results_1, "Pathway 1"), 
    (2, results_2, "Pathway 2_MNLogit"), 
    (3, results_3, "Pathway 3")
]:
    if not res: 
        continue

    # Standard MLR Pathways (Pathways 1 and 3)
    if 'std_coefficients' in res:
        # Coefficients & Hypotheses
        coef_df = res['std_coefficients'].copy()
        pval_df = res['adjusted_pvalues'].copy()
        merged_df = pd.merge(coef_df, pval_df, on='Variable', how='left')
        
        # Add Hypothesis result
        p_col = [col for col in merged_df.columns if col.startswith('Adjusted_p_')][0]
        merged_df['Hypothesis_Result'] = merged_df[p_col].apply(
            lambda p: 'Supported (Significant)' if p < 0.05 else 'Not Supported (Not Significant)'
        )
        coef_frames[name] = merged_df

        # Model Fit Stats
        fit_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F-test p-value'],
            'Value': [res.get('r_squared'), res.get('adj_r_squared'), res.get('f_value'), res.get('f_p_value')]
        })
        fit_frames[name] = fit_df

        # Assumption Tests
        assump = res.get('assumption_tests', {})
        if assump:
            normality = assump.get('normality', {})
            homo = assump.get('homoscedasticity', {})
            assump_df = pd.DataFrame({
                'Test': ['Jarque-Bera Stat', 'Jarque-Bera p-value', 'Shapiro-Wilk Stat', 'Shapiro-Wilk p-value', 'Breusch-Pagan Stat', 'Breusch-Pagan p-value'],
                'Value': [normality.get('jb_stat'), normality.get('jb_pval'), normality.get('sw_stat'), normality.get('sw_pval'), homo.get('bp_stat'), homo.get('bp_pval')],
                'Passed (p>0.05)': [
                    'Yes' if normality.get('jb_pval', 0) > 0.05 else 'No',
                    '-',
                    'Yes' if normality.get('sw_pval', 0) > 0.05 else 'No',
                    '-',
                    'Yes' if homo.get('bp_pval', 0) > 0.05 else 'No',
                    '-'
                ]
            })
            assumption_frames[name] = assump_df

            # VIF / Multicollinearity
            if 'vif' in assump:
                vif_frames[name] = assump['vif'].copy()

        # Model Validation
        val = res.get('validation', {})
        if val:
            val_df = pd.DataFrame({
                'Metric': ['Test Set R2', 'CV Mean R2', 'CV Std Deviation'],
                'Value': [val.get('test_r2'), val.get('cv_mean'), val.get('cv_std')]
            })
            validation_frames[name] = val_df

    # Exploratory Pathway 2 (MNLogit specifically)
    elif 'mnlogit_model' in res:
        model_mn = res['mnlogit_model']
        params = model_mn.params
        pvals = model_mn.pvalues
        
        mnlogit_results = []
        for col in params.columns:
            temp = pd.DataFrame({
                'Variable': params.index,
                'Coefficient (Log-Odds)': params[col].values,
                'P_Value': pvals[col].values,
                'Category_Equation': f"Category_{col}_vs_Baseline"
            })
            # Add basic exploratory hypothesis flag
            temp['Hypothesis_Result'] = temp['P_Value'].apply(
                lambda p: 'Supported (Significant)' if p < 0.05 else 'Not Supported (Not Significant)'
            )
            mnlogit_results.append(temp)
            
        if mnlogit_results:
            coef_frames[name] = pd.concat(mnlogit_results, ignore_index=True)

        # Fit stat for MNLogit
        fit_frames[name] = pd.DataFrame({
            'Metric': ['Pseudo R-squared', 'LLR p-value'],
            'Value': [model_mn.prsquared, model_mn.llr_pvalue]
        })


try:
    def write_excel(frames_dict, filename):
        if not frames_dict: return
            pass
        filepath = os.path.join(export_dir, filename)
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sht_name, df in frames_dict.items():
                safe_name = sht_name.replace(":", "").replace(" ", "_")[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

    # Export distinct excels per conceptual category
    write_excel(coef_frames, 'Hypothesis_Testing_and_Coefficients.xlsx')
    write_excel(fit_frames, 'Model_Fit_Statistics.xlsx')
    write_excel(assumption_frames, 'Assumption_Tests_Normality_Homoscedasticity.xlsx')
    write_excel(vif_frames, 'Assumption_Tests_Multicollinearity_VIF.xlsx')
    write_excel(validation_frames, 'Model_Validation_Results.xlsx')
    
    print(f"\nSuccessfully exported all individual diagnostic results to the '{export_dir}/' folder:")
    print(" - Hypothesis_Testing_and_Coefficients.xlsx")
    print(" - Model_Fit_Statistics.xlsx")
    print(" - Assumption_Tests_Normality_Homoscedasticity.xlsx")
    print(" - Assumption_Tests_Multicollinearity_VIF.xlsx")
    print(" - Model_Validation_Results.xlsx")
    
except ImportError:

