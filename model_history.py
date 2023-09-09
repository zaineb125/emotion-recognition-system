import matplotlib.pyplot as plt

def plot_model_history(model_history):
    # Extract training and validation loss for every 10 epochs
    training_loss = model_history.history['loss'][::10]
    validation_loss = model_history.history['val_loss'][::10]

    # Extract training and validation accuracy for every 10 epochs
    training_accuracy = model_history.history['accuracy'][::10]
    validation_accuracy = model_history.history['val_accuracy'][::10]

    # Get the number of epochs for the sampled points
    epochs = range(1, len(training_loss) * 10 + 1, 10)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, 'b', label='Training Loss')
    plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Set y-axis limits for loss
    plt.ylim(0, max(max(training_loss), max(validation_loss)) * 1.1)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Set y-axis limits for accuracy
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have trained your model and stored the history in `model_history`
# plot_model_history(model_history)
