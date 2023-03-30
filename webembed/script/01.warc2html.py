"""
This script extracts the HTML from the pages of commoncrawl.org
"""

from warcio.archiveiterator import ArchiveIterator
from os import mkdir
from os.path import join
import json

base = 'D:\\doc2vec\\dataset\\commoncrawl.org'
fil = 'CC-MAIN-20150323172132-{i}-ip-10-168-14-71.ec2.internal.warc'
output = 'html.{i}'

for i in range(40):
    with open(join(base, fil.format(i=str(i).zfill(5))), 'rb') as fp:
        j = 0
        output_path = join(base, output.format(i=i))
        mkdir(output_path)
        urls = {}
        for record in ArchiveIterator(fp):
            output_file = f'{i}.{j}.html'
            if record.rec_type == 'response':
                if record.http_headers.get_header('Content-Type') == 'text/html':
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    html = record.content_stream().read().decode('utf-8', errors='ignore')
                    with open(join(output_path, output_file), 'w') as wp:
                        wp.write(html)
                    urls[output_file] = url
                    j += 1
        with open(join(output_path, 'urls.json'), 'w') as wp:
            json.dump(urls, wp)
