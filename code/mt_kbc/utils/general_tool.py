import datetime
import codecs


def trace(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str, args)))


def dress(value, size=10):
    return (str(float(value))+'0'*size)[:size]


def read(file_path):
    with codecs.open(file_path, 'r', 'utf-8', errors='ignore') as read_f:
        for line in read_f:
            yield line.strip()
