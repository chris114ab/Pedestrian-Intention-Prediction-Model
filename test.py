from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    print(thresholds)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])

    plt.show()


# with open("data/scores.txt") as file:
#     probs = file.readline().strip("[|]\n").split(",")
#     labels = file.readline().strip("[|]\n").split(",")

def parse_float_arrays(input_str):
    # Find all floating point numbers in the input string
    float_strings = re.findall(r"\[\s*([\d\.e-]+)\s*\]", input_str)
    # Convert strings to floats
    float_values = [float(val) for val in float_strings]
    return np.array(float_values, dtype=np.float32)

# Function to parse boolean array from the input string
def parse_bool_array(input_str):
    # Split the string by spaces and filter out empty strings
    bool_strings = input_str.strip("[]").split()
    # Convert strings to booleans
    bool_values = [val == "True" for val in bool_strings]
    return np.array(bool_values, dtype=bool)

probs = """[array([0.9918942], dtype=float32), array([0.08325734], dtype=float32), array([0.9950269], dtype=float32), array([0.9994822], dtype=float32), array([0.9981982], dtype=float32), array([0.24327114], dtype=float32), array([0.6930465], dtype=float32), array([0.00149411], dtype=float32), array([0.99987984], dtype=float32), array([0.99998415], dtype=float32), array([0.8590882], dtype=float32), array([0.68448806], dtype=float32), array([0.9838215], dtype=float32), array([0.00018226], dtype=float32), array([0.99730814], dtype=float32), array([0.992109], dtype=float32), array([0.9512164], dtype=float32), array([0.2766029], dtype=float32), array([0.90248305], dtype=float32), array([0.99982315], dtype=float32), array([0.9535758], dtype=float32), array([0.99976224], dtype=float32), array([0.6335796], dtype=float32), array([0.99989176], dtype=float32), array([0.18754844], dtype=float32), array([0.78321415], dtype=float32), array([0.9935675], dtype=float32), array([0.99975103], dtype=float32), array([0.9998529], dtype=float32), array([0.9997819], dtype=float32), array([0.30513647], dtype=float32), array([0.9796306], dtype=float32), array([0.4659114], dtype=float32), array([0.882364], dtype=float32), array([0.978293], dtype=float32), array([0.00619808], dtype=float32), array([0.53233993], dtype=float32), array([0.92021465], dtype=float32), array([0.99995565], dtype=float32), array([0.99602306], dtype=float32), array([0.13293555], dtype=float32), array([0.96424305], dtype=float32), array([0.96573114], dtype=float32), array([0.00290183], dtype=float32), array([0.99998784], dtype=float32), array([0.01736439], dtype=float32), array([0.99965024], dtype=float32), array([0.9959539], dtype=float32), array([0.9549166], dtype=float32), array([0.99849343], dtype=float32), array([0.03708636], dtype=float32), array([0.05815809], dtype=float32), array([0.9993642], dtype=float32), array([0.9949686], dtype=float32), array([0.15851161], dtype=float32), array([0.9765633], dtype=float32), array([0.36534578], dtype=float32), array([0.1314324], dtype=float32), array([0.02295152], dtype=float32), array([0.99999976], dtype=float32), array([0.99999917], dtype=float32), array([0.00112242], dtype=float32), array([0.2602586], dtype=float32), array([0.91264385], dtype=float32), array([0.04270111], dtype=float32), array([0.99979895], dtype=float32), array([0.00416662], dtype=float32), array([0.9895661], dtype=float32), array([0.03576682], dtype=float32), array([0.83605325], dtype=float32), array([0.00602713], dtype=float32), array([0.99989283], dtype=float32), array([0.00197331], dtype=float32), array([0.9990169], dtype=float32), array([0.0198016], dtype=float32), array([0.9996495], dtype=float32), array([0.9887893], dtype=float32), array([0.25876045], dtype=float32), array([0.20957315], dtype=float32), array([0.89227], dtype=float32), array([0.9370489], dtype=float32), array([0.99999785], dtype=float32), array([0.15176333], dtype=float32), array([0.99570054], dtype=float32), array([0.3020015], dtype=float32), array([0.11906195], dtype=float32), array([0.35893187], dtype=float32), array([0.06001982], dtype=float32), array([0.9918276], dtype=float32), array([0.9924136], dtype=float32), array([0.01298828], dtype=float32), array([0.99335736], dtype=float32), array([0.02605197], dtype=float32), array([0.11622707], dtype=float32), array([0.21386094], dtype=float32), array([0.00090731], dtype=float32), array([0.04080795], dtype=float32), array([0.8315508], dtype=float32), array([0.9697999], dtype=float32), array([0.07665134], dtype=float32)]"""
labels = """[ True False False False False  True  True False  True  True False False
 False False False  True False  True  True  True  True False  True  True
 False False False  True  True False  True False  True False False False
  True False False False  True  True False False  True  True  True False
 False  True  True  True  True  True  True False False  True  True  True
  True False  True  True False  True False False  True False  True  True
 False False False False False  True  True False False False  True False
 False  True  True  True  True False  True False False False  True False
 False  True  True  True]"""
# Parse the arrays
float_array = parse_float_arrays(probs)
bool_array = parse_bool_array(labels)

labels = [1 if x else 0 for x in bool_array]
print(labels)
# print(probs)
# probs = [float(x) for x in probs]

# print(probs)
# print(labels)

plot_roc_curve(np.array(labels), np.array(float_array))
# plot_confusion_matrix(np.array(labels), np.array(float_array))
