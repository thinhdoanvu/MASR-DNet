from ultralytics import YOLO, RTDETR

if __name__ == "__main__":

    # model = RTDETR("cfg/models/rt-detr/rtdetr-resnet101.yaml")
    models = [
        # "cfg/models/v8/yolov8m.yaml",
        #"cfg/models/v8/yolov8m.yaml",
        #"cfg/models/v9/yolov9m.yaml",
        #"cfg/models/v9/yolov9m.yaml",
        #"cfg/models/v10/yolov10l.yaml",
        #"cfg/models/v10/yolov10l.yaml",
        #"cfg/models/11/yolo11l.yaml",
        #"cfg/models/11/yolo11l.yaml",
        #"cfg/models/v12/yolov12l.yaml",
        #"cfg/models/v12/yolov12l.yaml",
        "cfg/models/rt-detr/rtdetr-resnet101.yaml"
        ]

    # Train YOLO models for Ablation Studies
    for model_path in models:
        print(f"Training model: {model_path}") 
        model = RTDETR(model_path)
        model.train(
            data="../AICUP25/aortic_valve.yaml",
            epochs=100,
            batch=8,
            imgsz=960,
            patience = 20,
            device=0
        )
    
    
    # Chay riêng cho cua Thinh thoi
    # model = YOLO("cfg/models/abl/v4.yaml")

    # model.train(
    #     data="../AICUP25/aortic_valve.yaml",
    #     epochs=300,
    #     batch=8,
    #     imgsz=960,
    #     optimizer="SGD",  # đổi optimizer # DGS - Adam
    #     lr0=1e-4,  # SGD = 0.001, Adam = 1e-4
    #     weight_decay=0.01,  # weight decay
    #     warmup_epochs=10,  # Kéo dài warmup để ổn định DCNv2 và Contrastive
    #     patience = 30,  # Đừng tắt sớm, mô hình cần thời gian hồi phục sau mỗi cú nhảy múa
    #     box = 12,  # Tăng trọng số Box loss lên để ưu tiên học vị trí
    #     cls = 0.5,  # Giảm nhẹ trọng số Cls để nhường chỗ cho Box
    #     close_mosaic=15,  # Tắt mosaic sớm hơn để tinh chỉnh box vật thể nhỏ
    # )

    # H200
    # model.train(
    #     data="../AICUP25/aortic_valve.yaml",
    #     epochs=200,
    #     batch=64,  # 140GB VRAM cho phép đẩy lên 128 để gradient mượt nhất
    #     imgsz=960,
    #     workers=32,  # Chỉnh theo CPU của bạn
    #     optimizer="SGD",
    #     lr0=1e-4,
    #     weight_decay=0.05,  # Tăng một chút để tránh overfitting do model phức tạp
    #     warmup_epochs=10.0,  # Kéo dài warmup để ổn định DCNv2 và Contrastive
    #     cos_lr=True,  # Giảm LR theo hình sin giúp hội tụ sâu hơn
    #     label_smoothing=0.1,
    #     close_mosaic=15,  # Tắt mosaic sớm hơn để tinh chỉnh box vật thể nhỏ
    #     patience = 50,  # Đừng tắt sớm, mô hình cần thời gian hồi phục sau mỗi cú nhảy múa
    #     box = 7.5,  # Tăng trọng số Box loss lên để ưu tiên học vị trí
    #     cls = 0.5,  # Giảm nhẹ trọng số Cls để nhường chỗ cho Box
    # )

    # Tiếp tục huấn luyện (fine-tune) khi lo STOP
    # model.train(
    #     data="../AICUP25/aortic_valve.yaml",
    #     epochs=200,
    #     imgsz=640,
    #     batch=32,
    #     resume=True,  # False = khởi tạo lại optimizer, True = tiếp tục từ checkpoint
    #     pretrained=True  # ép dùng trọng số đã có
    # )
    #
