from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # build a new model from YAML

# Train the model
model.train(
    data='../FADS_EAS_Tree-Throw-Prediction/datasets5/TreeThrow.yaml', 
    epochs=600, 
    imgsz=400,
    batch=32,
    cache=True,
    name='all_data',
    augment=True,
    single_cls=True,
    box=7.5,
)