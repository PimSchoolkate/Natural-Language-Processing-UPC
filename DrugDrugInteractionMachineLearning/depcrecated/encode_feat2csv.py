import sys

from feature_collector import encode_train_test, feature_collector

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]

    collector = feature_collector()
    x, y, feature_names = collector.parse_stdin(sys.stdin)

    encode_train_test(train, test, path=f"./csv/")