from typing import List, Tuple
import warnings
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines import exceptions
from lifelines.statistics import multivariate_logrank_test
from lifelines.statistics import proportional_hazard_test

PALETTE_EST=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
              'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
              'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
              'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
PALETTE_CONF=['rgba(31, 119, 180, 0.4)', 'rgba(255, 127, 14, 0.4)',
               'rgba(44, 160, 44, 0.4)', 'rgba(214, 39, 40, 0.4)',
               'rgba(148, 103, 189, 0.4)', 'rgba(140, 86, 75, 0.4)',
               'rgba(227, 119, 194, 0.4)', 'rgba(127, 127, 127, 0.4)',
               'rgba(188, 189, 34, 0.4)', 'rgba(23, 190, 207, 0.4)']

def encode_predictor(df: pd.DataFrame, column: str, drop_level: str) -> pd.DataFrame:
    '''
    Encode a single categorical variable, dropping the specified level
    '''
    dummies = pd.get_dummies(df[column])
    if drop_level in dummies.columns:
        dummies = dummies.drop(drop_level, axis = 1)
    dummies.columns = [f"{column}_{col_name}" for col_name in dummies.columns]
    return dummies

def get_dummies(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    '''
    Get K - 1 dummy variables for each categorical variable in df.
    For each variable, the most frequent level is used as reference.
    Reference levels and level sample sizes returned as dicts
    '''
    df_list = []
    reference_levels = {}
    sample_size = {}
    for col in df:
        if is_string_dtype(df[col]):
            counts = df[col].value_counts()
            ref = counts.index[0]
            encoded = encode_predictor(df, col, ref)
            df_list.append(encoded)
            reference_levels[col] = ref
            sample_size[col] = counts.to_dict()
        else:
            df_list.append(df[col])
            sample_size[col] = {col: df[col].value_counts().sum()}
    return pd.concat(df_list, axis = 1), reference_levels, sample_size

def parse_cox_results(df: pd.DataFrame, refs: dict, ns: dict) -> pd.DataFrame:
    '''
    Process cph.summary.reset_index() (df) for forest plot
    '''
    cats = list(refs.keys())
    # Add columns for confidence intervals
    df['upper'] = list(df['exp(coef) upper 95%'] - df['exp(coef)'])
    df['lower'] = list(df['exp(coef)'] - df['exp(coef) lower 95%'])
    df = df[['covariate', 'exp(coef)', 'p', 'upper', 'lower']]
    # Dissect each covariate into "Predictor" and "Predictor Level" labels
    df['Predictor'] = df['covariate']
    df['Predictor Level'] = df['covariate']
    for cat in cats:
        idx = df['covariate'].str.contains(cat)
        level = df.loc[idx, 'covariate'].str.split(cat).str[1].str.lstrip('_')
        df.loc[idx, 'Predictor'] = cat
        df.loc[idx, 'Predictor Level'] = level
        df.loc[len(df.index)] = [f'{cat}_{refs[cat]}', 1, float('nan'), 0, 0, cat, refs[cat]]
    # Add sample sizes
    df['N'] = 0
    for var in ns:
        for level in ns[var]:
            idx = (df['Predictor'] == var) & (df['Predictor Level'] == level)
            df.loc[idx, 'N'] = ns[var][level]
    return df

def forest_plot(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    '''
    Produces Forest Plot for visualizing CoxPH results.
    Height of plot will scale with total number of predictor levels.
    '''
    levels = df['Predictor Level'].value_counts()
    labels = [f'{p["Predictor Level"]}  \t|\t  N = {p["N"]}  \t|\t  Reference'
              if p['p'] != p['p']
              else f'{p["Predictor Level"]}  \t|\t  N = {p["N"]}  \t|\t  <i>p</i>-value = {p["p"]:.2e}' for i,p in df.iterrows()]
    labels = [f'<b>{labels[i]}</b>' if p["p"] < 0.05 else f'{labels[i]}' for i,p in df.iterrows()]
    df['Label'] = labels
    fig = px.scatter(df,
                     x = "exp(coef)",
                     y = "Label",
                     color = "Predictor",
                     error_x = "upper",
                     error_x_minus = "lower",
                     labels = {"exp(coef)": "Estimated Hazard Ratio (95% CI)"})
    fig.update_traces(marker = dict(size = 10))
    fig.update_layout(title = 'Hazard Ratios',
                      font = dict(size = 14),
                      showlegend = True,
                      legend_title_text = "Predictors",
                      legend_traceorder = 'reversed',
                      legend = dict(
                        orientation = 'h',
                        yanchor = 'bottom',
                        xanchor = 'left',
                        x = 0,
                        y = 1,
                        font = dict(size = 14)),
                      margin = dict(l = 100, t = 100, r = 100),
                      height = 200 + 150*math.log(len(levels)))
    fig.update_yaxes(title_text = '')
    fig.add_vline(x = 1, line_dash = 'dot', opacity = 0.4)
    return fig

def prep_survival(df: pd.DataFrame,
                  tte: str,
                  event: str,
                  dichotomize: bool,
                  use_numeric_codings: bool) -> pd.DataFrame:
    '''
    Preps df for survival analysis (KM or CoxPH) by:
      1) Ensuring event is coded correctly (0 if censored, 1 if event occured)
      2) Ensuring only complete observations
      3) If dichotomize = True, dichotomizes numeric predictor variables (not tte or event) using mid-point
      4) If use_numeric_codings = True, string predictor variables will be treated as numeric
    
    Value: A prepped DF for KM or Cox PH analysis
    '''
    codings = {"alive": 0, "Alive": 0, "dead": 1, "Dead": 1, "deceased": 1, "Deceased": 1}
    df.replace({event: codings}, inplace = True)
    df.dropna(inplace = True)
    for col in df:
        if col == tte: continue
        if col == event: continue
    if dichotomize:
        for col in df:
            if col == tte: continue
            if col == event: continue
            if is_numeric_dtype(df[col]):
                labels, bins = pd.qcut(df[col], 2, labels = ['low', 'high'], retbins = True)
                labels_w_threshold = {'low': 'low (< {t:.3f})'.format(t = bins[1]),
                                      'high': 'high (> {t:.3f})'.format(t = bins[1])}
                df[col] = labels.replace(labels_w_threshold)
    if use_numeric_codings:
        for col in df:
            if col == tte: continue
            if col == event: continue
            if is_string_dtype(df[col]):
                df[col] = df[col].astype('category').cat.codes
    return df

def km_plot(df: pd.DataFrame,
            tte: str,
            event: str,
            predictor: str,
            max_levels: int = 10) -> go.Figure:
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    i = 0
    df_data = prep_survival(df[[tte, event, predictor]],
                            tte,
                            event,
                            dichotomize = True,
                            use_numeric_codings = False)
    if df_data.groupby(predictor).ngroups > max_levels:
        fig.add_annotation(text = "Can't generate plot with more than {levels} levels".format(levels = max_levels),
                           showarrow = False)
        fig.update_layout(title = predictor)
        return fig
    else:
        fm = multivariate_logrank_test(df_data[tte], df_data[predictor], df_data[event])
        stat = '$\chi_{degf}^2 = {stat:.{digits}f}$'.format(stat = fm.test_statistic,
                                                            degf = fm.degrees_of_freedom,
                                                            digits = 3)
        pval = '<i>p</i>-value = {pval:.{digits}e}'.format(pval = fm.p_value, digits = 3)
        for level, grouped_df in df_data.groupby(predictor):
            kmf.fit(grouped_df[tte], grouped_df[event])
            kmf_df = pd.DataFrame({"tte": kmf.survival_function_.index,
                                   "est": kmf.survival_function_['KM_estimate'],
                                   "upper_CI": kmf.confidence_interval_['KM_estimate_upper_0.95'],
                                   "lower_CI": kmf.confidence_interval_['KM_estimate_lower_0.95']})
            # Add point estimate
            fig.add_trace(go.Scatter(
                name = predictor + " = " + str(level),
                x = kmf_df['tte'],
                y = kmf_df['est'],
                mode = "lines",
                line = dict(color = PALETTE_EST[i]),
                hovertemplate = 'Survival probability = %{y:,.3f}'
            ))
            # Add upper 0.95 CI
            fig.add_trace(go.Scatter(
                name = "Upper 95% CI",
                x = kmf_df['tte'], 
                y = kmf_df['upper_CI'],
                mode = 'lines',
                line = dict(width = 0),
                showlegend = False,
                hoverinfo = 'skip'
            ))
            # Add lower 0.95 CI
            fig.add_trace(go.Scatter(
                name = "Lower 95% CI",
                x = kmf_df['tte'],
                y = kmf_df['lower_CI'],
                mode = 'lines',
                line = dict(width = 0),
                fill = 'tonexty',
                fillcolor = PALETTE_CONF[i],
                showlegend = False,
                hoverinfo = 'skip'
            ))
            # Add logrank test statistic and p-value
            fig.add_annotation(
                xref = 'x domain',
                yref = 'y domain',
                x = 0.05,
                y = 0.1,
                text = stat,
                showarrow = False,
                font = dict(size = 15)
            )
            fig.add_annotation(
                xref = 'x domain',
                yref = 'y domain',
                x = 0.05,
                y = 0.05,
                text = pval,
                showarrow = False,
                font = dict(size = 14)
            )

            # Add axis and hover styles
            fig.update_layout(
                title = predictor,
                xaxis_title = tte,
                yaxis_title = "Survival probability",
                hovermode = 'x unified',
                height = 700
            )

            i += 1
        return fig
    
# def coxph_analysis(df: pd.DataFrame,
#                    tte: str,
#                    event: str,
#                    predictors: List[str]) -> Tuple[dict, dict, go.Figure, dict, dict]:
#     pass

def coxph_analysis(df: pd.DataFrame,
                   tte: str,
                   event: str,
                   predictors: List[str]):
    df_data = prep_survival(df[[tte, event] + predictors],
                            tte,
                            event,
                            dichotomize = False,
                            use_numeric_codings = False)
    # Get model matrix. For each categorical variable, the reference level will
    #   be the most frequent
    df_model, refs, ns = get_dummies(df_data)
    # Fit model and get fit statistics
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col = tte, event_col = event)
    # warnings.simplefilter("always")
    # with warnings.catch_warnings(record = True) as warning:
        # convergence_warnings = []
        # convergence_errors = []
        # try:
            # cph.fit(df_model, duration_col = tte, event_col = event)
        # except exceptions.ConvergenceError as err:
            # convergence_errors = str(err).split(". ")
            # convergence_errors = []
            # return [], [], go.Figure, [], convergence_errors, []
        # finally:
            # for w in warning:
                # if w.category.__name__ == "ConvergenceWarning":
                #    convergence_warnings.append(str(w.message))
    model_fit = pd.DataFrame({
        "index": ["Baseline estimation",
                  "Number of observations",
                  "Number of events observed",
                  "Partial log-likelihood",
                  "Partial AIC",
                  "Concordance"],
        "value": [cph.baseline_estimation_method,
                  cph._n_examples,
                  cph.event_observed.sum(),
                  '{:.2f}'.format(cph.log_likelihood_),
                  '{:.2f}'.format(cph.AIC_partial_),
                  '{:.2f}'.format(cph.concordance_index_)]
    }).to_dict('records')
    # Get tabular estimates and forest plot
    estimates_df = cph.summary.reset_index()[['covariate', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    estimates_tab = estimates_df.to_dict('records')
    cox_df = parse_cox_results(estimates_df, refs, ns)
    forest = forest_plot(cox_df)
    # Get results of proportional hazard test
    diag = proportional_hazard_test(cph, df_model, time_transform = 'km')
    diag = diag.summary.reset_index()
    diag.rename(columns = {'index': 'coefficient'}, inplace = True)
    diag = diag.to_dict('records')
    return model_fit, estimates_tab, forest, diag