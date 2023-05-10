from ultralytics import YOLO

# load model
model = YOLO('runs/detect/yolov8m/weights/best.pt')

# eval model
metrics = model.val(
    data='../tree-throw-yolov8/dataset-train/TreeThrow.yaml',
    imgsz=400,
    mode='test',
    batch=32,
    cache=False,
    name='val/yolov8m',
    save_txt=True,
    save_conf=True,
    single_cls=True,
    conf=0.1
)

print(('%10s' * 5) % ('Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95'))
print(('%10.3f' * 5) % (metrics.box.mp, metrics.box.mr, metrics.box.mf1, metrics.box.map50, metrics.box.map))