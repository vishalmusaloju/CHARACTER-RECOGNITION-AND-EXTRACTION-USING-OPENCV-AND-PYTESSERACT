import os
import time
import moviepy as mp
import speech_recognition as sr
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import io
import cv2
from PIL import Image
import pytesseract
import numpy as np
from pytesseract import Output

# Set the path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract audio from video
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

# Function to split audio into chunks
def split_audio(audio_path, chunk_duration=30):
    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)
    chunks = []
    for i in range(0, duration_ms, chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunks.append(chunk)
    return chunks

# Function to transcribe audio chunk
def transcribe_audio_chunk(audio_chunk):
    r = sr.Recognizer()

    # Convert AudioSegment to a byte array
    audio_bytes = io.BytesIO()
    audio_chunk.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    with sr.AudioFile(audio_bytes) as source:
        audio = r.record(source)
        try:
            print(f"Transcribing audio chunk...")
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            print(f"Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"API request failed: {e}")
            return ""

# Function to process video and get transcriptions
def process_video_audio_transcription(video_path):
    # Extract audio from video
    audio_path = extract_audio(video_path)

    # Split audio into chunks
    chunks = split_audio(audio_path)

    # Using ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor() as executor:
        transcriptions = list(executor.map(transcribe_audio_chunk, chunks))

    # Combine the transcriptions and save the result
    full_transcription = "\n".join(transcriptions)
    with open("transcription_output.txt", "w") as file:
        file.write(full_transcription)
    print("Transcription saved to: transcription_output.txt")

# OCR on images
def process_image_ocr(image_path):
    img = np.array(Image.open(image_path))
    text = pytesseract.image_to_string(img)

    print("Text detected from image:")
    print(text)

    results = pytesseract.image_to_data(img, output_type=Output.DICT)

    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = int(float(results["conf"][i]))
        if conf > 40:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    cv2.imshow("Image OCR", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# OCR on video frames
def process_video_ocr(video_path, skip_frames=5, window_width=800, window_height=600):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames if needed
        if frame_count % skip_frames != 0:
            continue

        # Perform OCR on the frame
        text_full = pytesseract.image_to_string(frame)

        # Perform OCR with detailed results
        results = pytesseract.image_to_data(frame, output_type=Output.DICT)

        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            text_single = results["text"][i]
            conf = int(float(results["conf"][i]))
            if conf > 40:
                text_single = "".join([c if ord(c) < 128 else "" for c in text_single]).strip()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text_single, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 3)

        # Resize the frame to the desired window size
        frame = cv2.resize(frame, (window_width, window_height))

        # Display the frame with OCR results
        cv2.imshow("Video OCR", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Main function to execute the process
if __name__ == "__main__":
    print("Choose the OCR process you want to perform:")
    print("1. Image OCR")
    print("2. Video OCR (Frames)")
    print("3. Video Audio Transcription")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        image_path = input("Enter the path to the image file: ")
        process_image_ocr(image_path)

    elif choice == "2":
        video_path = input("Enter the path to the video file: ")
        process_video_ocr(video_path)

    elif choice == "3":
        video_path = input("Enter the path to the video file: ")
        process_video_audio_transcription(video_path)

    else:
        print("Invalid choice, please choose 1, 2, or 3.")
