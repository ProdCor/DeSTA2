from desta import DestaModel
import os
import glob
import csv
from tqdm import tqdm

# Configuration
DATASET_PATH = "./EMIS_dataset/"  # Change this to your dataset path
OUTPUT_CSV = "/path/to/your/results/"

# Load model
print("Loading model...")
model = DestaModel.from_pretrained("/path/to/desta2/model", local_files_only=True)
model.to("cuda")

# Find audio files
wav_files = glob.glob(os.path.join(DATASET_PATH, "**/*.wav"), recursive=True)
wav_files = list(set(wav_files))  # Remove duplicates
print(f"Found {len(wav_files)} audio files")

# Process files
results = []
for audio_file in tqdm(wav_files):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": audio_file},
            {"role": "user", "content": "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral."} #"Describe the emotion of the speaker in one word"} }
        ]
        
        generated_ids = model.chat(messages, max_new_tokens=32, do_sample=False)
        response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        emotion = response.strip().split()[-1].lower()
        
        results.append([os.path.basename(audio_file), emotion, response.strip()])
        
    except Exception as e:
        print(f"Error with {audio_file}: {e}")
        results.append([os.path.basename(audio_file), "ERROR", str(e)])

# Save results
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "predicted_emotion", "full_response"])
    writer.writerows(results)

print(f"Results saved to {OUTPUT_CSV}")