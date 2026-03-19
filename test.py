from functools import lru_cache
from pathlib import Path
import sys

# add path to sys
sys.path.append('e:/project lung disease/lung_disease')

from main import get_model

model = get_model()
print("Model created.")

print(f"Total layers: {len(model.layers)}")
last_conv_name = None
for layer in reversed(model.layers):
    try:
        sh = layer.output_shape
        if isinstance(sh, list):
            sh = sh[0]
        if len(sh) == 4:
            last_conv_name = layer.name
            break
    except Exception as e:
        print(f"Error on layer {layer.name}: {e}")

print(f"Detected last conv layer name: {last_conv_name}")
