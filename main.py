import gradio as gr
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to avoid CUDA driver issue

# Load and prepare data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train, 10)

class_names = ["Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
               "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]

# Create and train simple model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=3, batch_size=64, verbose=1)

# Función para predicción con depuración
def classify_image(img):
    try:
        print("Debug: Tipo de imagen de entrada:", type(img))
        if isinstance(img, np.ndarray):
            print("Debug: Forma del array de entrada:", img.shape)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] in [3, 4]:
                img = Image.fromarray(img, mode='RGB')
            elif len(img.shape) == 2:
                img = Image.fromarray(img, mode='L')
            else:
                raise ValueError(f"Forma de imagen no soportada: {img.shape}")
        elif not isinstance(img, Image.Image):
            raise ValueError(f"Tipo de imagen no soportado: {type(img)}")
        
        print("Debug: Tipo después de conversión:", type(img))
        print("Debug: Modo de la imagen PIL:", img.mode)
        
        img = img.convert('L').resize((28, 28))
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
        print("Debug: Forma del array procesado:", img_array.shape)
        
        pred = model.predict(img_array)
        print("Debug: Predicciones crudas:", pred)
        return {class_names[i]: float(pred[0][i]) for i in range(10)}
    except Exception as e:
        print("Error en classify_image:", str(e))
        raise

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),  # Remove 'shape' parameter
    outputs=gr.Label(num_top_classes=3),
    title="Clasificador Fashion-MNIST",
    description="Sube una imagen de ropa y el modelo predice la categoría."
)

iface.launch()