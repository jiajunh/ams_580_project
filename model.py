from sklearn.linear_model import LogisticRegression


def train_logistic_model(X_train, Y_train, args):
    model = LogisticRegression(random_state=args.seed,
                               max_iter=500,
                               penalty="l2",
                            #    C=1.0, 
                            #    l1_ratio=0.5,
                            #    solver="saga",
                               ).fit(X_train, Y_train)
    return model

