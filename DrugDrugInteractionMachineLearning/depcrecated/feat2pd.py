import sys

import pandas as pd

from feature_collector import feature_collector

if __name__ == '__main__':
    collector = feature_collector()
    x, y, feature_names = collector.parse_stdin(sys.stdin)

    xdf = pd.DataFrame(x, columns=feature_names, index=[f"{sys.argv[2]}_{i}" for i in range(len(x))])
    ydf = pd.DataFrame(y, columns=['Y'], index=[f"{sys.argv[2]}_{i}" for i in range(len(x))])

    partnull = 0

    for col in xdf.columns:
        nulls = xdf[xdf[col] == 'null'][col].count()
        partnull = partnull + nulls/xdf.shape[0]
        print(f"{col} \t has {nulls} \t null values, which is {nulls/xdf.shape[0]} \t% of the data")

    print(f"percentage of the data that is empty: {partnull/len(xdf.columns)}")
    xdf.to_csv(path_or_buf=f"{sys.argv[1]}_x.csv")
    ydf.to_csv(path_or_buf=f"{sys.argv[1]}_y.csv")