import os
from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt")  # Asegúrate de que el modelo es válido
    # model = YOLO("best.pt")  # Otra opción para cargar el modelo

    # Obtener la ruta del directorio actual del archivo .py
    dir = os.path.dirname(os.path.abspath(__file__))
    # Construir la ruta al archivo data.yaml en el dataset
    data_path = os.path.join(dir, 'Construction_toysv3iyolov11', 'data.yaml')

    # Entrenar el modelo e incluir parámetros adicionales para estabilidad
    train_results = model.train(
        data=data_path,
        epochs=100,
        # img_size=640,  # Descomentar si se necesita una resolución específica
        device='cpu'    # Cambiar a 'cuda' si tienes una GPU compatible
    )
    print("Training complete. Results:", train_results)

if __name__ == '__main__':
    train()
