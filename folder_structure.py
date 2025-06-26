'''
Code to automatically generate folder structures. You need to modify the file names to be reflective of what you need.
'''
import os

project_structure = {
    "emotion_classifier": {
        "configs": ["config.yaml"],
        "data": ["__init__.py", "downloader.py", "ravdess_dataset.py", "collate.py"],
        "models": ["__init__.py", "base_model.py", "attention_classifier.py", "hubert_model.py"],
        "train": ["__init__.py", "trainer.py", "evaluate.py"],
        "utils": ["__init__.py", "logger.py", "saver.py", "tb_writer.py", "plot_utils.py", "misc.py"],
        "runs": [],
        "checkpoints": [],
        "misclassified": [],
        "notebooks": ["main_experiments.ipynb"],
        "": ["main.py", "README.md", "requirements.txt"]
    }
}

def create_structure(base_path, structure):
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for subfolder, files in contents.items():
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder:
                os.makedirs(subfolder_path, exist_ok=True)
            for file in files:
                file_path = os.path.join(subfolder_path if subfolder else folder_path, file)
                if not os.path.exists(file_path):  # ✅ Skip if already exists
                    with open(file_path, "w") as f:
                        f.write(f"# Placeholder for {file}\n")
                else:
                    print(f"Skipping existing file: {file_path}")

create_structure(".", project_structure)
print("Project structure created (existing files left untouched).")
