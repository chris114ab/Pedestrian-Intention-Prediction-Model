from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
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
    plt.grid()
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
    bool_strings = input_str.strip("[]").split(",")
    # Convert strings to booleans
    bool_values = [val == "True" for val in bool_strings]
    return np.array(bool_values, dtype=bool)

probs = """[0.00606004],[0.00018098],[0.00131563],[0.9999976],[0.99990904],[0.8549511],[0.00339116],[0.99999595],[0.00174061],[0.98383707],[0.00013353],[0.62133807],[0.00023084],[0.9999914],[0.00401813],[0.07695097],[0.99999],[0.99998283],[0.9973463],[0.99256694],[0.38537723],[0.9999993],[0.00014474],[0.09493618],[0.00314835],[0.00124601],[0.9999999],[0.99868304],[9.2155795e-05],[0.98979175],[0.00023696],[0.00367901],[0.00010987],[1.],[0.99987626],[1.],[0.99986994],[0.9757751],[0.32415453],[0.6237629],[0.8492267],[0.9999925],[0.9999716],[0.99973315],[0.9999975],[7.998535e-06],[0.99995685],[7.226613e-05],[2.0252715e-05],[0.1479386],[0.00026316],[0.9952596],[1.],[0.92483246],[0.22218643],[0.99989295],[0.9986922],[0.95739853],[7.188028e-05],[0.9999409],[0.00333943],[0.9781843],[7.5059084e-05],[0.00169778],[0.06857612],[0.0084323],[0.96316993],[0.11914834],[5.3254756e-05],[0.88678306],[5.6969653e-05],[0.9997588],[0.01116939],[9.448721e-05],[0.99999857],[0.9983308],[0.9975556],[0.999946],[0.9996871],[0.0006044]"""
labels = """True,False,True,True,True,False,False,True,False,True,False,True,False,True,False,False,True,True,True,True,True,True,False,False,False,False,True,True,False,True,False,True,False,True,True,True,False,False,False,True,True,True,True,True,True,False,True,True,True,False,True,False,True,False,False,True,True,True,False,True,False,True,True,False,False,False,True,False,True,False,False,True,False,False,True,False,False,True,True,False"""
# Parse the arrays
float_array = parse_float_arrays(probs)
bool_array = parse_bool_array(labels)
print(len(float_array))
labels = [1 if x else 0 for x in bool_array]
print(labels)
# print(probs)
# probs = [float(x) for x in probs]

# print(probs)
# print(labels)


# plot_roc_curve(np.array(labels), np.array(float_array))

# plot_confusion_matrix(np.array(labels), np.array(float_array))


precision, recall, _ = precision_recall_curve(np.array(labels), np.array(float_array))
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.grid()
plt.show()