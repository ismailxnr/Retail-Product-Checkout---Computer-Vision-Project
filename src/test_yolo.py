from ultralytics import YOLO

def main():
    # ✅ Eğitim sonrası en iyi modeli yükle
    model = YOLO("runs/detect/rpc_val_eval_only2/weights/best.pt")

    # ✅ Test kümesini değerlendir ve detaylı sonuçları kaydet
    model.val(
        data="data.yaml",
        split="test",
        device=0,
        save_json=True,    # → COCO formatlı val_predictions.json üretir
        verbose=True       # → Tüm sınıflar için P, R, mAP çıktısı verir (console + results.txt)
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

