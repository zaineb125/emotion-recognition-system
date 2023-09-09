import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,precision_score
import seaborn as sns
from keras.models import load_model


def plot_confusion_matrix(model, validation_generator):
    class_names = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

    # Get true labels
    y_true = validation_generator.classes

    # Predict using the model
    y_pred = model.predict_generator(validation_generator).argmax(axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate precision for all classes
    overall_precision = precision_score(y_true, y_pred, average='micro')

    # Print overall precision
    print(f"Overall Precision: {overall_precision:.2f}")

    # Calculate accuracy for each class using (TP + TN) / (TP + TN + FP + FN)
    class_accuracy = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[i, :]) - tp
        fn = np.sum(cm[:, i]) - tp
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        class_accuracy.append(accuracy)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print accuracy for each class
    for i, class_name in enumerate(class_names):
        print(f"Accuracy for class '{class_name}': {class_accuracy[i]:.2f}")


# Example usage:
# Load the model and validation generator
# model = load_model("your_model.h5")
# validation_generator = ...

# Plot confusion matrix and calculate class-wise accuracy
# plot_confusion_matrix(model, validation_generator)
