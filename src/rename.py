import os
import soundfile as sf
from glob import glob

PROCESSED_PATH = "data/processed/"
HUMAN_DIR = os.path.join(PROCESSED_PATH, "human", "voice")
NONHUMAN_DIR = os.path.join(PROCESSED_PATH, "non_human")

def rename_human_files():
    print("Renaming human audio files...")
    files = glob(os.path.join(HUMAN_DIR, "*.wav"))
    for idx, filepath in enumerate(files):
        try:
            data, sr = sf.read(filepath)
            duration = int(len(data) / sr)
            new_name = f"audio_1_0_{duration}.wav"
            new_path = os.path.join(HUMAN_DIR, new_name)

            # Ensure no name clash
            count = 1
            while os.path.exists(new_path):
                new_name = f"audio_1_0_{duration}_{count}.wav"
                new_path = os.path.join(HUMAN_DIR, new_name)
                count += 1

            os.rename(filepath, new_path)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

def rename_nonhuman_files():
    print("Renaming non-human audio files...")
    classes = sorted(os.listdir(NONHUMAN_DIR))  # ensure consistent label IDs
    class_to_id = {cls: i for i, cls in enumerate(classes)}  # 0–10

    for cls in classes:
        cls_path = os.path.join(NONHUMAN_DIR, cls)
        files = glob(os.path.join(cls_path, "*.wav"))

        for idx, filepath in enumerate(files):
            try:
                data, sr = sf.read(filepath)
                duration = int(len(data) / sr)
                class_id = class_to_id[cls]
                new_name = f"audio_0_{class_id}_{duration}.wav"
                new_path = os.path.join(cls_path, new_name)

                # Ensure no name clash
                count = 1
                while os.path.exists(new_path):
                    new_name = f"audio_0_{class_id}_{duration}_{count}.wav"
                    new_path = os.path.join(cls_path, new_name)
                    count += 1

                os.rename(filepath, new_path)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    rename_human_files()
    rename_nonhuman_files()
    print("All files renamed.")
