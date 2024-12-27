#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
import time
import urllib.request
import wave
import numpy as np
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

SAMPLE_RATE = 16000
DETECTION_THRESHOLD = 0.8
COOLDOWN_DURATION = 3

def create_audio_stream(rate: int = SAMPLE_RATE) -> subprocess.Popen:
    return subprocess.Popen([
        'sox', '-d', '-t', 'raw', '-r', str(rate), '-b', '16', '-c', '1', '-e', 'signed-integer', '-',
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def read_audio_chunk(stream: subprocess.Popen, chunk_size: int = SAMPLE_RATE) -> np.ndarray:
    raw_data = stream.stdout.read(chunk_size * 2)  # 2 bytes per sample for int16
    return np.frombuffer(raw_data, dtype=np.int16)

def load_audio_file(filename: str) -> np.ndarray:
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with wave.open(filepath, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000:
            raise ValueError("Audio must be 16kHz mono")
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

def process_audio(interpreter: tf.lite.Interpreter, audio: np.ndarray) -> tuple[float, float]:
    audio = np.clip(audio.astype(np.float32) / 32768.0 * 32768, -32768, 32767).astype(np.int16)
    
    features = frontend_op.audio_microfrontend(
        tf.convert_to_tensor(audio),
        sample_rate=16000,
        window_size=30,
        window_step=20,
        num_channels=40,
        upper_band_limit=7500,
        lower_band_limit=125,
        enable_pcan=True,
        min_signal_remaining=0.05,
        out_scale=1,
        out_type=tf.uint16,
    ).numpy().astype(np.float32) * 0.0390625

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    is_quantized = input_details['dtype'] == np.int8
    
    predictions = []
    for i in range(len(features) - 2):
        chunk = np.reshape(features[i:i+3], (1, 3, 40))
        
        if is_quantized:
            input_scale = input_details['quantization_parameters']['scales'][0]
            input_zero_point = input_details['quantization_parameters']['zero_points'][0]
            chunk = (chunk / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details["index"], chunk)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details["index"])[0][0]
        if is_quantized:
            zero_point = output_details['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - zero_point) / 255.0
        
        predictions.append(output)
    
    return max(predictions), np.mean(predictions)

def continuous_detection(interpreter: tf.lite.Interpreter) -> None:
    print("\nListening for wake word... (Press Ctrl+C to stop)")
    
    audio_stream = create_audio_stream()
    audio_buffer = np.array([], dtype=np.int16)
    buffer_size = int(SAMPLE_RATE * 3)
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    last_dot_time = time.time()
    cooldown_until = 0
    
    try:
        while True:
            current_time = time.time()
            chunk = read_audio_chunk(audio_stream, chunk_size)
            if len(chunk) == 0:
                break

            audio_buffer = np.concatenate([audio_buffer, chunk])
            if current_time >= cooldown_until and current_time - last_dot_time >= 0.1:
                print(".", end="", flush=True)
                last_dot_time = current_time

            if len(audio_buffer) >= buffer_size:
                max_pred, _ = process_audio(interpreter, audio_buffer)
                if max_pred > DETECTION_THRESHOLD and current_time >= cooldown_until:
                    print(f"\nWake word detected! (confidence: {max_pred:.4f})")
                    cooldown_until = current_time + COOLDOWN_DURATION
                    last_dot_time = current_time

            if len(audio_buffer) > buffer_size:
                audio_buffer = audio_buffer[-buffer_size:]
    finally:
        audio_stream.terminate()

def process_file(interpreter: tf.lite.Interpreter, filename: str) -> None:
    audio_data = load_audio_file(filename)
    max_pred, avg_pred = process_audio(interpreter, audio_data)
    
    print(f"\nProcessing {filename}:")
    print(f"Max prediction: {max_pred:.4f}")
    print(f"Average prediction: {avg_pred:.4f}")
    print(f"Wake word {'detected' if max_pred > 0.5 else 'not detected'}")

def download_default_model(model_name: str) -> None:
    print(f"Downloading default model {model_name}...")
    url = "https://github.com/esphome/micro-wake-word-models/raw/refs/heads/main/models/v2/okay_nabu.tflite"
    urllib.request.urlretrieve(url, model_name)
    print("Download complete")

def load_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main() -> None:
    parser = argparse.ArgumentParser(description='Wake word detector')
    parser.add_argument('file', nargs='?', help='Audio file to process (optional)')
    parser.add_argument('--model', type=str, default='okay_nabu.tflite', help='Path to TFLite model file')
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if args.model == 'okay_nabu.tflite' and not os.path.exists(model_path):
        download_default_model(model_path)
    interpreter = load_model(model_path)
    
    try:
        if args.file:
            process_file(interpreter, args.file)
        else:
            continuous_detection(interpreter)
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
