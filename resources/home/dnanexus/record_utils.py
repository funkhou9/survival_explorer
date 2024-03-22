import os
from io import StringIO
import dxpy
import subprocess
import pandas as pd
from typing import List, Tuple, Literal

def get_dataset_or_cohort_id(dxid: dict) -> str:
    '''
    Params:
      dxid - A DNAnexus link {'$dnanexus_link': 'record-XXXX'} to a Cohort or Dashboard record
    Value:
      <project-ID>:<record-ID> of Cohort or Dataset to be used in `dx extract_dataset`
    '''
    dxrecord = dxpy.DXRecord(dxid)
    types = dxrecord.describe()['types']
    details = dxrecord.get_details()
    if "DashboardView" in types:
        record_id = dxpy.PROJECT_CONTEXT_ID + ":" + details['dataset']['$dnanexus_link']
    elif "CohortBrowser" in types or "DatabaseQuery" in types:
        record_id = dxpy.PROJECT_CONTEXT_ID + ":" + dxrecord.get_id()
    else:
        raise Exception("Input record not recognized as either Cohort or Dashboard")
    return record_id

def get_data_dict(dxid: dict) -> Tuple[str, pd.DataFrame]:
    ''' 
    Params:
      dxid - A DNAnexus link {'$dnanexus_link': 'record-XXXX'} to a Cohort or Dashboard record
    Value:
      Tuple containing:
          1) <project-ID>:<record-ID> of Cohort or Dataset to be used in `dx extract_dataset`
          2) Apollo data dictionary
    '''
    record_name = dxpy.DXRecord(dxid).describe()['name']
    record_id = get_dataset_or_cohort_id(dxid)
    cmd = ['dx', 'extract_dataset', record_id, '-ddd']
    subprocess.check_call(cmd)
    dd_df = pd.read_csv(record_name + ".data_dictionary.csv")
    os.remove(record_name + ".data_dictionary.csv")
    os.remove(record_name + ".codings.csv")
    os.remove(record_name + ".entity_dictionary.csv")
    return record_id, dd_df

def get_field_metadata(dxid: dict, _use_tiles: bool = True) -> Tuple[str, str, List[dict]]:
    '''
    Params:
      dxid - A DNAnexus link {'$dnanexus_link': 'record-XXXX'} to a Cohort or Dashboard record
      _use_tiles - Whether to get metadata from fields represented in Dashboard Tiles.
          If not, gets metadata from primary entity fields only.
    Value:
      Tuple containing:
          1) Name of record object
          1) <project-ID>:<record-ID> of Cohort or Dataset to be used in `dx extract_dataset`
          2) List of field metadata, obtained from data dictionary. If 'use_tiles' = True,
             then metadata from all fields represented in Dashboard Tiles will be obtained.
             Additionally, the following "tile metadata" will be obtained:
                 "tile_id"
                 "tile_type"
                 "tile_coordinate"
    '''
    details = dxpy.DXRecord(dxid).get_details()
    record_name = details['name']
    record_id, dd_df = get_data_dict(dxid)

    # Two "containers" should be present within detials,
    #   one for Dashboard Tiles, another for Data Preview section of CB
    containers = details['dashboardConfig']['cohort_browser']['containers']
    ted_container = next((i for i in containers if i['id'] == 'ted_container'), None)
    dashboard_tiles = next((i for i in containers if i['id'] == 'dashboard_tiles'), None)

    # Assemble metadata
    patient = ted_container['tiles'][0]
    entity, field = list(patient['dataQuery']['fields'])[0].split("$")
    primary_entity = dd_df[(dd_df['entity'] == entity)]
    # Always get metadata from global primary key
    meta = primary_entity[(primary_entity['name'] == field)].to_dict('records')
    meta[0]['tile_id'] = None
    meta[0]['tile_type'] = None
    meta[0]['tile_coordinate'] = None
    if (not _use_tiles) or (len(dashboard_tiles['tiles']) == 0) :
        # If not using tiles (or no tiles exist in record), just assemble metadata from primary entity
        p_entity_meta = primary_entity[(primary_entity['name'] != field)].to_dict('records')
        for field in p_entity_meta:
            field['tile_id'] = None
            field['tile_type'] = None
            field['tile_coordinate'] = None
        return record_name, record_id, (meta + p_entity_meta)
    else:
        # Add metadata for each field represented in Dashboard Tiles
        for query in dashboard_tiles['tiles']:
            tile_id = query['id']
            tile_type = query['type']
            for key,val in query['dataQuery']['fields'].items():
                entity, field = val.split("$")
                field_meta = dd_df[(dd_df['entity'] == entity) & (dd_df['name'] == field)].to_dict('records')[0]
                field_meta['tile_id'] = tile_id
                field_meta['tile_type'] = tile_type
                field_meta['tile_coordinate'] = key
                meta.append(field_meta)
        return record_name, record_id, meta

class Record():
    def __init__(self, dxid: dict, use_tiles: bool = True):
        self.record_name, self.record_id, self.field_metadata = get_field_metadata(dxid, _use_tiles = use_tiles)

    def extract_data(self,
                     tile_config: Literal['survival'] = 'survival',
                     use_titles: bool = True,
                     add_units: bool = False,
                     id_as_index: bool = True) -> pd.DataFrame:
        '''
        Extracts data for analysis
        Params:
          use_titles - Whether to use field titles (instead of field names) as column names
          add_units - Whether to append units to column names "<column name> (<unit>)"
          tile_config - Specifies field order:
              'survival': Puts fields of tile_type = "SurvivalPlot" first (if present)
          id_as_index - Whether to use patient IDs as row index
        Value:
          DataFrame with extracted data
        '''
        # Retrieve important info from each field
        entities = [field['entity'] for field in self.field_metadata]
        fields = [field['name'] for field in self.field_metadata]
        titles = [field['title'] for field in self.field_metadata]
        units = [field['units'] for field in self.field_metadata]
        colnames = [e + "." + f for e, f in zip(entities, fields)]
        for idx,meta in enumerate(self.field_metadata):
            meta['colname'] = colnames[idx]

        # Retrieve data
        colnames_str = ','.join(colnames)
        cmd = ['dx', 'extract_dataset', self.record_id, '--fields', colnames_str, '-o', '-']
        data = subprocess.run(cmd, stdout = subprocess.PIPE).stdout.decode()
        data_df = pd.read_csv(StringIO(data))

        # Modify data_df according to args
        if tile_config == 'survival':
            if 'SurvivalPlot' in [field['tile_type'] for field in self.field_metadata]:
                patient_col = [self.field_metadata[0]['colname']]
                tte_col = [field['colname'] for field in self.field_metadata if (field['tile_type'] == 'SurvivalPlot') & (field['tile_coordinate'] == 'x')]
                event_col = [field['colname'] for field in self.field_metadata if (field['tile_type'] == 'SurvivalPlot') & (field['tile_coordinate'] == 'y')]
                remaining_cols = list(set([field['colname'] for field in self.field_metadata]) - set(patient_col + tte_col + event_col))
                data_df = data_df[patient_col + tte_col + event_col + remaining_cols]
                # Enforce only one record per patient
                data_df.drop_duplicates(subset = patient_col, inplace = True, ignore_index = True)


        if use_titles:
            data_df.rename(columns = dict(zip(colnames, titles)), inplace = True)
            colnames = titles
        
        if add_units:
            unit_labels = [f'{i} ({j})' for i,j in zip(colnames, units)]
            data_df.rename(columns = dict(zip(colnames, unit_labels)), inplace = True)

        if id_as_index:
            data_df = data_df.set_index(data_df.columns[0])
            data_df.index.name = None

        return data_df
