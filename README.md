# microWakeWord Python Inference Demo

microWakeWord is a TensorFlow-based wake word detection framework. This allows devices to listen for certain words or phrases (like 'Alexa', or 'Okay Nabu'). It creates small but effective models, suitable for microcontrollers and other resource-constrained devices.

This demo shows how to run microWakeWord v2 models in Python. You can use either your microphone or provide a WAV file.

By default, it'll automatically download the 'Okay Nabu' model. You can download other pre-trained microWakeWord models from [esphome/micro-wake-word-models](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2), and then use the `--model` argument to use them.

## Quick start

1. Clone this repository
2. Run `python setup.py`
3. Start the script:
   - In live mode (uses your microphone, requires sox): `python script.py`
   - In file mode: `python script.py sample.wav` (files must be 16kHz mono WAV format)

### Additional Options

`--model`: Specify a different TFLite model file (default: 'okay_nabu.tflite')
```bash
python script.py --model path/to/model.tflite
```
