import gzip
datasetPath='/public/share/datasets/';
f = gzip.open(datasetPath+'orkut-groups.txt.gz', 'rb')
file_content = f.read()
f.close()
