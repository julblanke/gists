import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('var True')
    plt.xlabel('var Predicted')

# Example usage
sensitivity = 0.65
specificity = 0.85

tn = 100 * specificity  # true negative
fp = 100 - tn  # false positive
fn = 100 * (1 - sensitivity)  # false negative
tp = 100 - fn  # true positive

cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(cm, classes=['Negative', 'Positive'], normalize=True)
fig.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
