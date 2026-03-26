from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO(f"C:/Users/VU/Documents/OBD/v3_H200_backup/v4/v4_640_94.0_66.9/weights/best.pt")  # load a custom model
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category
