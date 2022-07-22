#from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class feature_collector():
    def __init__(self, label_idx=3):
        self.feature_names = {}
        self.next_idx = 0
        self.label_idx = label_idx
        self.x = []
        self.y = []

    def parse_line_into_dict(self, line):
        feat = {}
        y = None
        for i, item in enumerate(line.split('\t')):
            if i < self.label_idx:
                continue
            elif i == self.label_idx:
                y = str(item)
                # self.y.append(str(item))
            else:
                if "=" in item:
                    key, value = item.split("=")
                    if "\n" in value:
                        value = value.replace("\n", "")
                else:
                    continue
                if key in feat.keys():
                    if type(feat[key]) == list:
                        feat[key].append(value)
                    else:
                        feat[key] = [feat[key], value]
                else:
                    feat[key] = value
        # self.x.append(feat)
        return feat, y

    def parse_stdin_into_dict(self, stdin):
        for line in stdin:
            feat, y = self.parse_line_into_dict(line)
            self.x.append(feat)
            self.y.append(y)
        return self.x, self.y

    def parse_line_into_array(self, line):
        pair = {}
        for i, item in enumerate(line.split('\t')):
            if i < self.label_idx:
                continue
            elif i == self.label_idx:
                self.y.append(str(item))
            else:
                if "=" in item:
                    key, value = item.split("=")
                else:
                    continue
                if key not in self.feature_names.keys():
                    self.feature_names[key] = self.next_idx
                    self.next_idx = self.next_idx + 1
                pair[key] = value

        feat = ['null'] * (self.next_idx)
        for key, value in pair.items():
            feat[self.feature_names[key]] = value
        self.x.append(feat)

    def fix_length_x(self):
        for i, x in enumerate(self.x):
            if len(x) != self.next_idx :
                zeros = ['null'] * (self.next_idx)
                self.x[i] = x + zeros[len(x):]

    def parse_stdin_into_array(self, stdin):
        for line in stdin:
            self.parse_line_into_array(line)
        self.fix_length_x()
        return self.x, self.y, list(self.feature_names.keys())




def merge_train_test(train, test):
    train_df = pd.DataFrame(train, index=[f"train_{i}" for i in range(len(train))])
    test_df = pd.DataFrame(test, index=[f"test_{i}" for i in range(len(test))])
    return pd.concat([train_df, test_df])


def split_train_test(df):
    return df[["train" in idx for idx in list(df.index)]], df[["test" in idx for idx in list(df.index)]]


def convert_to_one_hot(data):
    return pd.get_dummies(data)


def convert_to_factors(data):
    return pd.DataFrame({col: pd.factorize(data[col])[0] for col in data.columns}, index=data.index)


def df2csv(df, path):
    df.to_csv(path=path)

def encode_train_test(train, test, path):
    data = merge_train_test(train, test)
    oh = convert_to_one_hot(data)
    fa = convert_to_factors(data)
    train_oh, test_oh = split_train_test(oh)
    train_fa, test_fa = split_train_test(fa)

    df2csv(train_oh, f"{path}train_oh")
    df2csv(test_oh, f"{path}test_oh")
    df2csv(train_fa, f"{path}train_fa")
    df2csv(test_fa, f"{path}test_fa")


