import sys
from feature_collector import feature_collector
from model_builder import model

if __name__ == "__main__":
    collector = feature_collector()

    x, y = collector.parse_stdin_into_dict(sys.stdin)

    model = model(mode="TRAIN", mod=sys.argv[1])
    model.train(x, y)
    model.pickle_model_and_encoder(sys.argv[1])