#!/usr/bin/env python3

import sys
from model_builder import model
from feature_collector import feature_collector

if __name__ == '__main__':
    collector = feature_collector()

    # load leaned model
    model = model(mode="PREDICT", name=sys.argv[1])


    for line in sys.stdin:
        x, y = collector.parse_line_into_dict(line)

        fields = line.strip('\n').split("\t")
        (sid, e1, e2) = fields[0:3]
        prediction = model.predict(x)[0]

        if prediction != "null":
            print(sid, e1, e2, prediction, sep="|")