from ultralytics import YOLO

# load model
model = YOLO('runs/detect/all_data2/weights/best.pt')

# eval model
metrics = model.val(
    data='../FADS_EAS_Tree-Throw-Prediction/datasets5/TreeThrow.yaml',
    imgsz=400,
    mode='test',
    batch=32,
    cache=False,
    name='val/all_data',
    save_txt=True,
    save_conf=True,
    single_cls=True,
    conf=0.1
)

# print(('%10s' * 5) % ('Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95'))
# print(('%10.3f' * 5) % (metrics.box.mp, metrics.box.mr, metrics.box.mf1, metrics.box.map50, metrics.box.map))