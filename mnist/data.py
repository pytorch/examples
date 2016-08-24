import os
import torch
import json
import errno
import codecs
from tqdm import tqdm

if not os.path.exists('data/raw/train-images-idx3-ubyte'):
    try:
        os.makedirs(os.path.join('data', 'raw'))
        os.makedirs(os.path.join('data', 'processed'))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    from six.moves import urllib
    import gzip
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    for url in urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join('data', 'raw', filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
             gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [b for b in tqdm(data[8:], total=length)]
        assert len(labels) == length
        return torch.LongTensor(labels)

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in tqdm(range(length)):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(data[idx])
                    idx += 1
        assert len(images) == length
        out = torch.FloatTensor(images)
        return out

print("Loading training set")
training_set = (
    read_image_file('data/raw/train-images-idx3-ubyte'),
    read_label_file('data/raw/train-labels-idx1-ubyte'),
)
print("Saving training set")
with open('data/processed/training.pt', 'wb') as f:
    torch.save(training_set, f)

print("Loading test set")
test_set = (
    read_image_file('data/raw/t10k-images-idx3-ubyte'),
    read_label_file('data/raw/t10k-labels-idx1-ubyte')
)
print("Saving test set")
with open('data/processed/test.pt', 'wb') as f:
    torch.save(test_set, f)
