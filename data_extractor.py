import json
from codecs import open as copen

def get_data(fname):
    inp = copen(fname, encoding='utf-8')
    data = inp.read()
    data = json.loads(data)
    inp.close()
    return data

def get_nice_data(fname):
    jdata = get_data(fname)
    texts = []
    opinions = []

    for q in jdata:
        texts.append(q['text'])
        curop = []

        for answer in q['answers']:
            for feature in answer.keys():
                if feature != 'text':
                    curop.append((answer['text'], feature))

        opinions.append(curop)

    return (texts, opinions)



