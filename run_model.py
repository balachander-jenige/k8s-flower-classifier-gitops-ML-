import argparse
import json
from pathlib import Path
import numpy as np
import joblib

MODEL_PATH = Path("artifacts/model.pkl")


def load_model(model_path=MODEL_PATH):
  model_file = Path(model_path)
  if not model_file.exists():
    raise FileNotFoundError(f"Model file not found at {model_file}. Please train the model first.")
  return joblib.load(model_file)


def parse_features(raw_input):
  value = raw_input.strip()
  if not value:
    raise ValueError("Input is empty.")

  try:
    data = json.loads(value)
  except json.JSONDecodeError:
    input_path = Path(value)
    if not input_path.exists():
      raise ValueError("Input must be valid JSON or a path to a JSON file.")
    with input_path.open("r") as f:
      data = json.load(f)

  if isinstance(data, dict):
    features = list(data.values())
  elif isinstance(data, list):
    features = data
  else:
    raise ValueError("Input JSON must be a list of numeric values or a dict of feature names to values.")

  features_array = np.asarray(features, dtype=float)
  if features_array.size != 4:
    raise ValueError(f"Expected 4 features, got {features_array.size}.")

  return features_array.reshape(1, -1)


def main():
  parser = argparse.ArgumentParser(description="Run the trained model on input data.")
  parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Inline JSON array like [5.1, 3.5, 1.4, 0.2] or a path to a JSON file."
  )
  args = parser.parse_args()

  try:
    X = parse_features(args.input)
    model = load_model(MODEL_PATH)
    prediction = model.predict(X)
    print(f"Prediction: {prediction[0]}")
  except Exception as e:
    raise ValueError(f"Error processing input: {e}") from e


if __name__ == "__main__":
  main()