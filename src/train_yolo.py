from ultralytics import YOLO

def main():
    # ✅ Detection modeli yükleniyor (segmentasyon değil!)
    model = YOLO("yolov8m.pt")

    # ✅ Eğitim başlat
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=12,
        workers=4,
        name="rpc_val_eval_only",
        device=0,
        amp=True,
        val=True,
        save=True
    )

    # ✅ Test değerlendirme
    model.val(data="data.yaml", split="test", device=0)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
