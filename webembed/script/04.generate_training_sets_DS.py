"""
This script generates the training set for the pages in DS (Data Set)
"""

from os.path import join
from os import mkdir, walk
import json
import pandas as pd
from tqdm import tqdm

all_html_path = 'D:\\doc2vec\\dataset\\all_html'
output_path = 'D:\\doc2vec\\dataset\\training_sets'
# .csv, .json exported from DS.db
db_test_set_path = 'D:\\doc2vec\\dataset\\DS.csv'
db_test_set_path2 = 'D:\\doc2vec\\dataset\\DS.json'

try:
    mkdir(output_path)
except:
    pass
#  df per threshold set
df_load = pd.read_csv(db_test_set_path)
df = df_load[(df_load.HUMAN_CLASSIFICATION != -1)]
df['state1_content'] = None
df['state1_tags'] = None
df['state1_content_tags'] = None
df['state2_content'] = None
df['state2_tags'] = None
df['state2_content_tags'] = None

with open(db_test_set_path2, 'r') as fp:
    db_test_set = json.load(fp)

# generate train set model
db_test_set_optimized = {}
for row in db_test_set:
    key1 = f"{row['appname']}.{row['state1']}"
    key2 = f"{row['appname']}.{row['state2']}"

    if key1 not in db_test_set_optimized:
        db_test_set_optimized[key1] = []
    if key2 not in db_test_set_optimized:
        db_test_set_optimized[key2] = []
    db_test_set_optimized[key1].append(row)
    db_test_set_optimized[key2].append(row)

content_train_model_set = {}
tags_train_model_set = {}
content_tags_train_model_set = {}

app_names = next(walk(all_html_path))[1]
pbar = tqdm(total=len(app_names))
for app_name in app_names:
    app_path = join(all_html_path, app_name)

    content_train_model_set[app_name] = []
    tags_train_model_set[app_name] = []
    content_tags_train_model_set[app_name] = []

    for file_name in next(walk(app_path))[2]:
        x = file_name.split('.')
        is_content = x[-1] == 'content'
        is_tags = x[-1] == 'tags'
        is_content_tags = x[-1] == 'content_tags'
        state_name = '.'.join(x[:-2] if is_content or is_tags or is_content_tags else x[:-1])

        key = f"{app_name}.{state_name}"
        result = key in db_test_set_optimized
        data = {
            'file_name': file_name,
            'state_name': state_name,
            'app_name': app_name,
            'path': join(app_name, file_name)
        }
        if not result:
            # add to model train set (leave out annotated human stuff in DS)
            if is_content:
                content_train_model_set[app_name].append(data)
            elif is_tags:
                tags_train_model_set[app_name].append(data)
            elif is_content_tags:
                content_tags_train_model_set[app_name].append(data)
        else:
            # add to threeshold df
            if is_content:
                df.loc[(df.appname == app_name) & (df.state1 == state_name), 'state1_content'] = json.dumps(data)
                df.loc[(df.appname == app_name) & (df.state2 == state_name), 'state2_content'] = json.dumps(data)
            if is_tags:
                df.loc[(df.appname == app_name) & (df.state1 == state_name), 'state1_tags'] = json.dumps(data)
                df.loc[(df.appname == app_name) & (df.state2 == state_name), 'state2_tags'] = json.dumps(data)
            if is_content_tags:
                df.loc[(df.appname == app_name) & (df.state1 == state_name), 'state1_content_tags'] = json.dumps(data)
                df.loc[(df.appname == app_name) & (df.state2 == state_name), 'state2_content_tags'] = json.dumps(data)

    pbar.update()

# save df, save content_train..
df.to_csv(join(output_path, 'DS_threshold_set.csv'))
with open(join(output_path, 'DS_content_tags_model_train_set.json'), 'w') as fp:
    json.dump(content_tags_train_model_set, fp)
with open(join(output_path, 'DS_tags_model_train_set.json'), 'w') as fp:
    json.dump(tags_train_model_set, fp)
with open(join(output_path, 'DS_content_model_train_set.json'), 'w') as fp:
    json.dump(content_train_model_set, fp)
