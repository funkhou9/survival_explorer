#!/usr/bin/env python3

import dxpy
import survival_explorer
import pandas as pd
from record_utils import Record

@dxpy.entry_point('main')
def main(record_dxlink, combine_cohorts):

    print("Extracting data from Apollo object(s)...")
    app_inputs = {}
    df_inputs = []
    record_inputs = [] 
    for link in record_dxlink:
        record = Record(link)
        df = record.extract_data(id_as_index = False)
        filename = record.record_name + '.csv'
        if combine_cohorts:
            df['cohort'] = record.record_name
            df_inputs.append(df)
            record_inputs.append(record.record_id)
        else:
            df.to_csv(filename)
            app_inputs[filename] = {'id': record.record_id, 'name': record.record_name}

    if combine_cohorts:
        df_combined = pd.concat(df_inputs, ignore_index = True)
        df_combined.to_csv("combined.csv")
        app_inputs["combined.csv"] = {'id': "+".join(record_inputs), 'name': 'combined'}

    app = survival_explorer.create_app(app_inputs)
    dxhandler = dxpy.get_handler(dxpy.JOB_ID)
    dxhandler.set_properties({"httpsAppState": "running"})
    app.run_server(host='0.0.0.0', port=443)

    return 1

dxpy.run()