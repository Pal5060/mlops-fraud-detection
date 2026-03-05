from sklearn.model_selection import train_test_split
from fraud_detection.config.config import load_config
from fraud_detection.data.loader import load_data
from fraud_detection.data.preprocessing import preprocess
from fraud_detection.models.train import train_model
from fraud_detection.models.evaluate import evaluate

def run():
    config = load_config()
    df = load_data(config["data"]["raw_path"])
    df = preprocess(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    model = train_model(X_train, y_train, config)
    score = evaluate(model, X_test, y_test)

    print(f"Accuracy: {score}")

if __name__ == "__main__":
    run()