"""Upload beer pong dataset to Hugging Face."""

from huggingface_hub import HfApi

api = HfApi()

repo_id = "pwalker/beer-pong-synthetic"

# Create the repo if it doesn't exist
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

# Upload the full dataset directory
print(f"Uploading /workspace/beer_pong_dataset to {repo_id}...")
api.upload_folder(
    folder_path="/workspace/beer_pong_dataset",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload synthetic beer pong dataset (1000 scenes x 4 views = 4000 images)",
)
print("Done!")
