import pandas as pd
import pylatex as pl
from typing import List
import plotly.graph_objects as go

def write_survival_report(header: dict,
                          data: dict,
                          cox: dict,
                          km: List[dict]) -> None:
    '''
    Generates a PDF report for a survival analysis consisting of three sections:
        - Data used
        - Cox PH results
        - Kaplan-Meier results

    Parameters:
        header: dict with keys
            "title": str
            "author": str
        data: dict with keys
            "record_id": str
            "data": dict
            "tte": str
            "event": str
            "predictors": List[str]
        cox: dict with keys
            "model_fit": dict
            "estimates": dict
            "plot": dict
            "diagnostics": dict
        km: List of dicts

    Returns: None
    Generates .pdf and .tex files
    '''
    # Initialize document
    doc = pl.Document()
    doc.packages.append(pl.Package('adjustbox'))
    doc.packages.append(pl.Package('geometry'))

    # Write preamble
    doc.preamble.append(pl.NoEscape(r'\geometry{a4paper, total={170mm,257mm}, left=20mm, top=20mm}'))
    doc.preamble.append(pl.Command('title', header['title']))
    doc.preamble.append(pl.Command('author', header['author']))
    doc.preamble.append(pl.Command('date', pl.NoEscape(r'\today')))
    doc.append(pl.NoEscape(r'\maketitle'))

    # Write data section
    data_df = pd.DataFrame.from_records(data['data'])
    N = data_df.shape[0]
    df_preview = data_df[data['data_cols']].head(20)
    with doc.create(pl.Section("Input Data")):
        with doc.create(pl.Subsection("Summary")) as summ:
            summ.append('Record ID: {id}'.format(id = data['record_id']))
            summ.append('\nTime-to-event field: {tte}'.format(tte = data['tte']))
            summ.append('\nEvent field: {event}'.format(event = data['event']))
            summ.append('\nPredictor fields: {pred}'.format(pred = ', '.join(data['predictors'])))
            summ.append('\nSample size: {N}'.format(N = N))
        with doc.create(pl.Subsection("Data Preview")) as dp:
            dp.append('First 20 participant records shown\n\n')
            dp.append(pl.NoEscape(r'\begin{adjustbox}{width=1\textwidth}'))
            dp.append(pl.NoEscape(df_preview.to_latex(escape = True)))
            dp.append(pl.NoEscape(r'\end{adjustbox}'))
    doc.append(pl.NoEscape(r'\newpage'))

    # Write Cox PH section
    model_df = pd.DataFrame.from_records(cox['model_fit'])
    model_df.columns = ['', '']
    estimates_df = pd.DataFrame.from_records(cox['estimates'])
    go.Figure(cox['plot']).write_image("cox_plot.png", width = 1000, height = 600, scale = 10)
    with doc.create(pl.Section("Cox PH Results")):
        with doc.create(pl.Subsection("Model Fit")) as model:
            model.append(pl.NoEscape(model_df.to_latex(index = False)))
        with doc.create(pl.Subsection("Estimates")) as estimates:
            estimates.append(pl.NoEscape(estimates_df.to_latex(escape = True,
                                                               index = False,
                                                               formatters = {'p': '{:,.2e}'.format})))
            with doc.create(pl.Figure(position='!htb')) as plot:
                plot.add_image("cox_plot.png", width = pl.NoEscape(r'0.95\linewidth'))
    doc.append(pl.NoEscape(r'\newpage'))
        
    # Write KM section:
    with doc.create(pl.Section("Kaplan-Meier Plots")):
        for i, km_dict in enumerate(km):
            go.Figure(km_dict).write_image("km_plot_{}.png".format(i), width = 1000, height = 600, scale = 10)
            with doc.create(pl.Figure(position='!htb')) as plot:
                plot.add_image("km_plot_{}.png".format(i), width = pl.NoEscape(r'0.95\linewidth'))
    
    # Generate report
    doc.generate_pdf("survival_report",
                     clean_tex = False,
                     compiler_args = ['-f'])