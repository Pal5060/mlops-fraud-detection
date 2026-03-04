from sklearn.linear_model import LogisticRegression

def train_model(X, y, config):
    model = LogisticRegression(max_iter=config["training"]["max_iter"])
    model.fit(X, y)
    return model