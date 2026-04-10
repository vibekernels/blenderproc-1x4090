"""Upload trained YOLOv8 cup detector to Hugging Face."""

from huggingface_hub import HfApi

api = HfApi()
repo_id = "pwalker/beer-pong-yolov8"

api.create_repo(repo_id, repo_type="model", exist_ok=True)

print(f"Uploading /workspace/beer_pong_yolo/runs/cup_detector_s to {repo_id}...")
api.upload_folder(
    folder_path="/workspace/beer_pong_yolo/runs/cup_detector_s",
    repo_id=repo_id,
    repo_type="model",
    commit_message="YOLOv8s fine-tuned 150 epochs on 4000 synthetic beer pong images (mAP50=0.912, mAP50-95=0.727)",
)
print("Done!")
