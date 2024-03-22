from dash import Dash, dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import pandas as pd
import plotly.graph_objects as go
import survival_utils as surv
import survival_report as report
import subprocess
import lifelines
from lifelines import exceptions
import warnings
from record_utils import Record

def create_app(datasets: dict):

    input_files = list(datasets.keys())
    default_file = input_files[0]
    default_df = pd.read_csv(default_file, index_col = 0)

    app = Dash(external_stylesheets = [dbc.themes.MATERIA])

    header = html.Div()
    
    controls = html.Div(
        [
            dbc.Card(
                [   
                    html.H4("Select data", style = {'color': '#1582B1'}, className = "card-title"),
                    html.Div(
                        [
                            dls.ClimbingBox(
                                dcc.Dropdown(
                                    id = "data_selection",
                                    options = [
                                        {"label": record['name'], "value": file} for file,record in datasets.items()
                                    ],
                                    value = default_file,
                                    clearable = False
                                ),
                                fullscreen = True
                            )

                        ]
                    ),
                    html.Div(
                        [
                            dbc.Input(placeholder = "record-XXXX", type = "text", id = 'new_data', debounce = True),
                            dbc.FormText("Add data by record-ID")
                        ]
                    )
                ],
                body = True
            ),
            html.Hr(style = {'borderColor': '#1582B1', 'opacity': 'unset'}),
            dbc.Card(
                [
                    html.H4("Select model parameters", style = {'color': '#1582B1'}, className = "card-title"),
                    html.Div(
                        [
                            dbc.Label("Select time column"),
                            dcc.Dropdown(
                                id = "tte_col",
                                options = [
                                    {"label": col, "value": col} for col in default_df.columns[1:]
                                ],
                                value = default_df.columns[1],
                                clearable = False
                            )
                        ],
                        style = {'marginBottom': 20}
                    ),
                    html.Div(
                        [
                            dbc.Label("Select event column"),
                            dcc.Dropdown(
                                id = "event_col",
                                options = [
                                    {"label": col, "value": col} for col in default_df.columns[1:]
                                ],
                                value = default_df.columns[2],
                                clearable = False
                            )
                        ],
                        style = {'marginBottom': 20}
                    ),
                    html.Div(
                        [
                            dbc.Label("Select predictor(s)"),
                            dbc.Checklist(
                                id = "predictor_cols",
                                options = [
                                    {"label": col, "value": col} for col in default_df.columns[1:]
                                ],
                                switch = True,
                                value = []
                            )
                        ]
                    )
                ],
                body = True
            )
        ]
    )
                        
    content = html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(label = "View Data", tab_id = "tab-data"),
                    dbc.Tab(label = "Cox Proportional Hazards", tab_id = "tab-cox"),
                    dbc.Tab(label = "Kaplan Meier Plots", tab_id = "tab-km")
                ],
                id = "tabs",
                active_tab = "tab-data"
            ),
            dls.Beat(html.Div(id = "tab-content"))
        ]
    )

    header = html.Div(
        html.H1("Survival Explorer",
                style = {'paddingTop': '1rem',
                         'paddingBottom': '0px',
                         'marginBottom': '0px'})
    )

    button = html.Div(
        dbc.Button("Download results to DNAnexus", 
                   id = 'download_button',
                   n_clicks = 0,
                   style = {'marginTop': '1rem'})
    )

    diaglog = html.Div(
        dcc.ConfirmDialog(id = 'confirm_box',
                          message = "Report Uploaded to DNAnexus in /survival_reports/")
    )

    app.layout = dbc.Container(
        [
            dls.ClimbingBox(diaglog, fullscreen = True),
            dls.Beat(dcc.Store(id = 'data_store', data = datasets)),
            dls.Beat(dcc.Store(id = 'results_store')),
            dbc.Row(
                [
                    dbc.Col(header, width = 'auto'),
                    dbc.Col(button, width = 'auto')
                ],
                justify = 'between'
            ),
            dbc.Row(
                [
                    dbc.Col(html.Hr(style = {'borderWidth': '0.25vh', 'borderColor': '#1582B1', 'opacity': 'unset'}))
                ],
                align = 'top'
            ),
            dbc.Row(
                [
                    dbc.Col(controls, md = 3),
                    dbc.Col(content, md = 9)
                ]
            )

        ],
        fluid = True
    )

    @app.callback(Output('data_selection', 'options'),
                  Output('data_store', 'data'),
                  Input('new_data', 'value'),
                  State('data_selection', 'options'),
                  State('data_store', 'data'),
                  prevent_initial_call = True)
    def update_data(record_id, options, data):
        record = Record({'$dnanexus_link': record_id})
        df = record.extract_data(id_as_index = False)
        filename = record.record_name + '.csv'
        df.to_csv(filename)
        options.append({'label': record.record_name,
                        'value': filename})
        data.append({filename: {'id': record_id, 'name': record.record_name}})
        return options, data

    @app.callback(Output('tte_col', 'options'),
                  Output('tte_col', 'value'),
                  Output('event_col', 'options'),
                  Output('event_col', 'value'),
                  Output('predictor_cols', 'options', allow_duplicate = True),
                  Output('predictor_cols', 'value'),
                  Input('data_selection', 'value'),
                  prevent_initial_call = True)
    def update_controls(file):
        df = pd.read_csv(file, index_col = 0)
        cols = df.columns
        return cols, cols[1], cols, cols[2], cols[1:], []

    @app.callback(Output("predictor_cols", "options"),
                  Input('data_selection', 'value'),
                  Input("tte_col", "value"),
                  Input("event_col", "value"))
    def limit_inputs(file, t, e):
        df = pd.read_csv(file, index_col = 0)
        lhs = [t, e]
        return [
            {"label": col, "value": col, "disabled": col in lhs}
            for col in df.columns[1:]
        ]

    @app.callback(Output('results_store', 'data'),
                  Input('data_selection', 'value'),
                  Input('tte_col', 'value'),
                  Input('event_col', 'value'),
                  Input('predictor_cols', 'value'),
                  State('data_store', 'data'))
    def fit_model(file, tte, event, predictors, record):
        df = pd.read_csv(file, index_col = 0)
        record_id = record[file]['id']
        if len(predictors) == 0:
            return {"data": df.to_dict('records'),
                    "record_id": record_id,
                    "data_cols": df.columns,
                    "tte": tte,
                    "event": event,
                    "predictors": predictors,
                    "km": [go.Figure()],
                    "cox": {"model_fit": [],
                            "estimates": [],
                            "plot": go.Figure(),
                            "diagnostics": [],
                            "fm_warnings": []}}
        # Subset observations
        id = df.columns[0]
        df_obs = df[[id, tte, event] + predictors].dropna()
        # Make KM Plots
        km = [surv.km_plot(df, tte, event, p) for p in predictors]
        # Fit Cox PH model
        warnings.simplefilter("always")
        with warnings.catch_warnings(record = True) as warning:
            model_fit, estimates, diag = [], [], []
            fm_warnings = []
            try:
                model_fit, estimates, plot, diag = surv.coxph_analysis(df, tte, event, predictors)
            except exceptions.ConvergenceError as err:
                plot = go.Figure()
                plot.add_annotation(text = '''
                                            Convergence failed due to matrix inversion problems
                                            ''',
                                    showarrow = False)
            except ZeroDivisionError as zerr:
                plot = go.Figure()
                plot.add_annotation(text = '''
                                            No participants available with complete records for all chosen predictors
                                            ''',
                                    showarrow = False)
            finally:
                for w in warning:
                    if w.category.__name__ == "ConvergenceWarning":
                        print(w.category.__name__)
                        print(w.message)
                        fm_warnings.append(w.category.__name__)
        return {"data": df_obs.to_dict('records'),
                "record_id": record_id,
                "data_cols": df_obs.columns,
                "tte": tte,
                "event": event,
                "predictors": predictors,
                "km": km,
                "cox": {"model_fit": model_fit,
                        "estimates": estimates,
                        "plot": plot,
                        "diagnostics": diag,
                        "fm_warnings": fm_warnings}}

    @app.callback(Output('confirm_box', 'displayed'),
                  Input('download_button', 'n_clicks'),
                  State('results_store', 'data'))
    def download_results(download_button, data):
        if download_button > 0:
            username = subprocess.check_output(['dx', 'whoami']).decode()
            try:
                report.write_survival_report(header = {'title': 'Survival Report', 'author': username},
                                             data = data,
                                             cox = data['cox'],
                                             km = data['km'])
            except subprocess.CalledProcessError as e:
                print("Errors occured during LaTeX compilation, although PDF report may have still been generated")
            cmd = "dx upload survival_report.pdf --path $DX_PROJECT_CONTEXT_ID:/survival_reports/"
            subprocess.check_call(cmd, shell = True)
            return True
        
    @app.callback(Output('tab-content', 'children'),
                  Input('tabs', 'active_tab'),
                  Input('results_store', 'data'))
    def render_content(active_tab, data):
        if active_tab == 'tab-data':
            nobs = len(data['data'])
            data_preview = html.Div(
                [
                    html.P(f'Sample size: {nobs}', style = {'margin': '1rem'}),
                    dash_table.DataTable(data['data'],
                                        [{"name": i, "id": i} for i in data['data_cols']],
                                        style_header = {'fontWeight': 'bold'},
                                        style_as_list_view = False,
                                        style_cell = {'textAlign': 'left', 'fontFamily': 'Helvetica'},
                                        page_size = 20)
                ]
            )
            return data_preview
        elif active_tab == 'tab-km':
            return [dcc.Graph(figure = plot, mathjax = True) for plot in data['km']]
        elif active_tab == 'tab-cox':
            modelfit_table = dash_table.DataTable(
                                    data['cox']['model_fit'],
                                    columns = [
                                        {
                                            "id": "index",
                                            "name": "_"
                                        },
                                        {
                                            "id": "value",
                                            "name": "_"
                                        }
                                    ],
                                    style_header = {'display': 'none'},
                                    style_as_list_view = True,
                                    style_cell_conditional = [{'if': {'column_id': 'index'},
                                                               'textAlign': 'left'}],
                                    style_cell = {'fontFamily': 'Helvetica', 'fontSize': '18px'}
                                )
            cox_content = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Collapse(
                                    [
                                        html.H5("Convergence Warning"),
                                        html.P('''
                                               Convergence warning(s) occured during the fitting of the
                                               cox proportional hazards model. Estimates of hazard
                                               ratios should be interpreted with caution.
                                               
                                               It is likely that one or more chosen predictors perfectly predicts event status or that multiple
                                               chosen predictors are highly correlated with one another.

                                               Please see job logs for more information. Search for 'ConvergenceWarning'.
                                               ''')
                                    ],
                                        id = "warnings",
                                        is_open = "ConvergenceWarning" in data['cox']['fm_warnings']
                                )
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(figure = data['cox']['plot'])
                                # dash_table.DataTable(
                                #     data['cox']['estimates'],
                                #     columns = [
                                #         {
                                #             "id": "covariate",
                                #             "name": "Predictor"
                                #         },
                                #         {
                                #             "id": "exp(coef)",
                                #             "name": "Hazard Ratio",
                                #             "type": "numeric",
                                #             "format": Format(precision = 3, scheme = Scheme.fixed)
                                #         },
                                #         {
                                #             "id": "exp(coef) lower 95%",
                                #             "name": "Lower 95% CI",
                                #             "type": "numeric",
                                #             "format": Format(precision = 3, scheme = Scheme.fixed)
                                #         },
                                #         {
                                #             "id": "exp(coef) upper 95%",
                                #             "name": "Upper 95% CI",
                                #             "type": "numeric",
                                #             "format": Format(precision = 3, scheme = Scheme.fixed)
                                #         },
                                #         {
                                #             "id": "p",
                                #             "name": "p-value",
                                #             "type": "numeric",
                                #             "format": Format(precision = 3, scheme = Scheme.exponent)
                                #         }
                                #     ],
                                #     style_header = {'fontWeight': 'bold'},
                                #     style_as_list_view = True,
                                #     style_cell = {'textAlign': 'left', 'fontFamily': 'Helvetica'}
                                # ),
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(modelfit_table, width = {'size': 6, 'offset': 3})
                        ]
                    )
                ]
            )
                        # dbc.Col(
                        #     html.Div(
                        #         [
                        #             html.Label("Proportional hazard assumption tests", style = {'textDecoration': 'underline'}),
                        #             dash_table.DataTable(
                        #                 data['cox']['diagnostics'],
                        #                 columns = [
                        #                     {
                        #                         "id": "coefficient",
                        #                         "name": "coefficient"
                        #                     },
                        #                     {
                        #                         "id": "test_statistic",
                        #                         "name": "test statistic",
                        #                         "type": "numeric",
                        #                         "format": Format(precision = 3, scheme = Scheme.fixed)
                        #                     },
                        #                     {
                        #                         "id": "p",
                        #                         "name": "p-value",
                        #                         "type": "numeric",
                        #                         "format": Format(precision = 3, scheme = Scheme.exponent)
                        #                     }
                        #                 ],
                        #                 style_data = {'border': 'none', 'textAlign': 'left'},
                        #                 style_header = {'fontWeight': 'bold', 'border': 'none', 'textAlign': 'left'},
                        #                 style_cell = {'fontFamily': 'Helvetica',
                        #                               'fontSize': '14px',
                        #                               'paddingTop': '0px',
                        #                               'paddingBottom': '0px'},
                        #                 fill_width = False
                        #             )
                        #         ]
                        #     ),
                        #     width = 3
                        # )


        return cox_content


    return app

