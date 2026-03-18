"""
Fine-tune YOLOv8 on the beer pong cup dataset and evaluate on test set.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-dir', default='/workspace/beer_pong_yolo')
    parser.add_argument('--model', default='yolov8n.pt', help='Base model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', default='/workspace/beer_pong_yolo/runs')
    parser.add_argument('--name', default='cup_detector')
    args = parser.parse_args()

    dataset_yaml = f"{args.yolo_dir}/dataset.yaml"

    # Load pretrained YOLOv8
    model = YOLO(args.model)

    # Train
    print(f"Training {args.model} for {args.epochs} epochs...")
    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=0,
        workers=4,
        patience=10,
        save=True,
        plots=True,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    best_model = YOLO(f"{args.project}/{args.name}/weights/best.pt")
    metrics = best_model.val(
        data=dataset_yaml,
        split='test',
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=f"{args.name}_test_eval",
        device=0,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
