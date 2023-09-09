from keras.utils import plot_model

def plot_my_model(model):
    plot_model(model, show_shapes=True,
               show_layer_names=False,
               expand_nested=True,
               rankdir="TB",
               dpi=100)