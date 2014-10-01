import json
from codecs import open as copen

def get_data(fname):
    inp = copen(fname, encoding='utf-8')
    data = inp.read()
    data = json.loads(data)
    inp.close()
    return data

def get_nice_data(jdata):
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

def clean_answer(answer, prologue, code):
    ret = []
    outp = copen('outp', code, encoding='utf-8')
    outp.write(prologue + '\n')
    for feature in answer:
        outp.write(feature[0] + ' ' + feature[1] + '\n')
    return ret
