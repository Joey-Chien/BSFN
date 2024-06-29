from datasets import load_dataset
import os

dataset = load_dataset("comHannah/bokeh-dataset")

save_dir = "bokeh_image"
os.makedirs(save_dir, exist_ok=True)

for i, item in enumerate(dataset['train']):
    original_filename = f"image_{i}_0.jpg"
    bokeh_filename = f"image_{i}_1.jpg"

    original_path = os.path.join(save_dir, original_filename)
    bokeh_path = os.path.join(save_dir, bokeh_filename)
    item['original_image'].save(original_path)
    item['bokeh_image'].save(bokeh_path)

    print(f"Saved {original_path} and {bokeh_path}")