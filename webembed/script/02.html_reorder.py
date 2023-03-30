"""
This script moves the pages of commoncrawl.org and DS_Crawls into the same folder all_html
"""

from os.path import join
from os import mkdir, walk
import shutil

ds_crawl_path = 'D:\\doc2vec\\dataset\\DS_Crawls'
commoncrawl_path = 'D:\\doc2vec\\dataset\\commoncrawl.org'
output_dataset = 'D:\\doc2vec\\dataset\\all_html'

mkdir(output_dataset)

#  get html in DS_Crawls and move them
for app_name in next(walk(ds_crawl_path))[1]:
    output_app_name = join(output_dataset, app_name)
    mkdir(output_app_name)
    html_path = join(ds_crawl_path, app_name, 'crawl0', 'doms')
    states = [x for x in next(walk(html_path))[2] if 'state' in x or 'index' in x]
    for state in states:
        input_state_path = join(html_path, state)
        output_state_path = join(output_app_name, state)
        shutil.copyfile(input_state_path, output_state_path)

# get htmls in commoncrawl.org and move
for app_name in next(walk(commoncrawl_path))[1]:
    better_app_name = f'commoncrawl_{app_name}'
    output_app_name = join(output_dataset, better_app_name)
    mkdir(output_app_name)
    html_path = join(commoncrawl_path, app_name)
    htmls = next(walk(html_path))[2]
    for html in htmls:
        input_html_path = join(html_path, html)
        output_html_path = join(output_app_name, html)
        shutil.copyfile(input_html_path, output_html_path)
