from datasets import load_dataset
dataset = load_dataset("PSewmuthu/Emotion_Video_Facial_Landmarks")
dataset["train"].to_csv("emotion_frames.csv")
print("✅ Dataset saved as emotion_frames.csv")
