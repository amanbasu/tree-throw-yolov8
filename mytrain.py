from ultralytics import YOLO
import mlflow

parent_run = mlflow.start_run()
print(f'run_id: {parent_run.info.run_id}')
# Load a model
model = YOLO('yolov8l.pt')  # build a new model from YAML
# Train the model
model.train(
    data='../tree-throw-yolov8/dataset-train/TreeThrow.yaml', 
    epochs=600, 
    imgsz=400,
    batch=32,
    cache=True,
    name='mlflow',
    augment=True,
    single_cls=True,
    box=0.05,
)
mlflow.end_run()

