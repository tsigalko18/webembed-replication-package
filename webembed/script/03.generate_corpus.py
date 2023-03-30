"""
This script extracts the tags, content, tags+content from the pages of all_html (data set DS + commoncrawl.org)
"""

from concurrent.futures import process
from os.path import join, exists
from os import walk
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Comment
import gensim
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import sys

#  fix maximum recursion depth exceeded while calling a Python object
sys.setrecursionlimit(10000)

corpus_path = 'D:\\doc2vec\\dataset\\all_html'


def html_structure_content(bs, corpus):
    try:
        if type(bs) == NavigableString:
            tokens = gensim.utils.simple_preprocess(bs.string)
            if len(tokens) > 0:
                corpus[0].extend(tokens)
                corpus[2].extend(tokens)
            return

        bs_has_name = bs.name != None
        bs_is_single_tag = str(bs)[-2:] == '/>'

        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'<{bs.name}>')
            corpus[2].append(f'<{bs.name}>')
        elif bs_has_name and bs_is_single_tag:
            corpus[1].append(f'<{bs.name}/>')
            corpus[2].append(f'<{bs.name}/>')
        try:
            for c in bs.children:
                if type(c) == Comment:
                    continue
                html_structure_content(c, corpus)
        except Exception as e:
            pass
        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'</{bs.name}>')
            corpus[2].append(f'</{bs.name}>')
    except Exception as e:
        print('html structure content error', e)
        pass


tasks = next(walk(corpus_path))[1]
tasks.sort()


def process_app(app_name):
    # print(app_name)
    app_path = join(corpus_path, app_name)
    html_files = [x for x in next(walk(app_path))[2] if 'html' == x.split('.')[-1]]
    for html in html_files:
        try:
            html_path = join(app_path, html)
            content_path = join(app_path, html + '.content')
            tags_path = join(app_path, html + '.tags')
            content_tags_path = join(app_path, html + '.content_tags')

            if exists(content_path) or exists(tags_path) or exists(content_tags_path):
                continue

            with open(html_path, 'r') as html_fd:
                raw = html_fd.read()
            soup = BeautifulSoup(raw, 'html.parser')
            corpus = ([], [], [])
            html_structure_content(soup, corpus)
            if len(corpus[0]) > 0:
                with open(content_path, 'w') as fd:
                    json.dump(corpus[0], fd)
            if len(corpus[1]) > 0:
                with open(tags_path, 'w') as fd:
                    json.dump(corpus[1], fd)
            if len(corpus[2]) > 0:
                with open(content_tags_path, 'w') as fd:
                    json.dump(corpus[2], fd)
        except:
            print('error in', app_name, html)


if __name__ == '__main__':
    p = Pool(cpu_count() - 1)
    with p:
        with tqdm(total=len(tasks)) as pbar:
            for i, _ in enumerate(p.imap_unordered(process_app, tasks)):
                pbar.update()
