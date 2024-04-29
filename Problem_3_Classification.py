from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import numpy as np
from tqdm import tqdm
from functions import io

# trainset
def load_trainset(embeddings):
    x_train, y_train = [], []
    with open("antonym-synonym set/Antonym_vietnamese.txt", "r", encoding="utf8") as f:
        ant_pairs = f.readlines()
    with open("antonym-synonym set/Synonym_vietnamese.txt", "r", encoding="utf8") as f:
        syn_pairs = f.readlines()

    # antonym data, y=0
    for pair in ant_pairs:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1 + v2)
        y_train.append(0)

    # synomyn data, y=1
    for pair in syn_pairs:
        words = pair.split()
        u1 = words[0].strip()
        try:
            u2 = words[1].strip()  # có dòng chỉ có 1 từ
        except:
            continue
        else:
            u2 = u2

        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1 + v2)
        y_train.append(1)

    return x_train, y_train

# testset
def load_testset(embeddings):
    x_test, y_test = [], []
    with open("datasets/ViCon-400/400_noun_pairs.txt", "r", encoding="utf8") as f:
        noun_pairs = f.readlines()
    with open("datasets/ViCon-400/400_verb_pairs.txt", "r", encoding="utf8") as f:
        verb_pairs = f.readlines()
    with open("datasets/ViCon-400/600_adj_pairs.txt", "r", encoding="utf8") as f:
        adj_pairs = f.readlines()
    testset = noun_pairs[1:] + verb_pairs[1:] + adj_pairs[1:]

    for pair in testset:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_test.append(v1 + v2)
        if words[2] == "ANT":
            y_test.append(0)
        else:
            y_test.append(1)

    return x_test, y_test


# load embeddings
def read_embedding(path):
    words = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines[2:]):
        line = line[:-1].strip().split(" ")  # remove "\n" last line

        line.remove("")
        words[line[0]] = np.array(list(map(float, line[1:])))
    return words


def main():
    # load embeddings, trainset, testset
    embeddings = read_embedding("word2vec/W2V_150.txt")
    X_train, y_train = load_trainset(embeddings)
    X_test, y_test = load_testset(embeddings)

    # logistic regression train
    print("Train with logistic regression ")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test
    pred_lr = model.predict(X_test)
    print("Precision score:", precision_score(y_test, pred_lr))
    print("Recall score:", recall_score(y_test, pred_lr))
    print("F1 score:", f1_score(y_test, pred_lr))
    print("Accuracy:", accuracy_score(y_test, pred_lr))

    print()
    print("Train with MLP")
    clf = MLPClassifier(hidden_layer_sizes=(500)).fit(X_train, y_train)
    pred_mlp = clf.predict(X_test)
    print("Precision score:", precision_score(y_test, pred_mlp))
    print("Recall score:", recall_score(y_test, pred_mlp))
    print("F1 score:", f1_score(y_test, pred_mlp))
    print("Accuracy:", accuracy_score(y_test, pred_mlp))

    output = []

    # Thêm thông tin về mô hình Logistic Regression
    output.append("Train with Logistic Regression\n")
    output.append(f"Precision score: {precision_score(y_test, pred_lr)}\n")
    output.append(f"Recall score: {recall_score(y_test, pred_lr)}\n")
    output.append(f"F1 score: {f1_score(y_test, pred_lr)}\n")
    output.append(f"Accuracy: {accuracy_score(y_test, pred_lr)}\n")

    # Thêm thông tin về mô hình MLP
    output.append("\nTrain with MLP\n")
    output.append(f"Precision score: {precision_score(y_test, pred_mlp)}\n")
    output.append(f"Recall score: {recall_score(y_test, pred_mlp)}\n")
    output.append(f"F1 score: {f1_score(y_test, pred_mlp)}\n")
    output.append(f"Accuracy: {accuracy_score(y_test, pred_mlp)}\n")
    
    Y_hat = clf.predict(X_test)
    print(classification_report(y_test, Y_hat, target_names=['synonym','antonym'], digits=4))
    output.append(f"\nClassification report:\n{classification_report(y_test, Y_hat, target_names=['synonym','antonym'], digits=4)}")
    io.write_output("Problem_3_result.txt", output) 
if __name__ == "__main__":
    main()
