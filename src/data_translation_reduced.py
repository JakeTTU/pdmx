# In[]
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# In[]
# Transformation #1: Values, value changes, changes of value changes
def transformation_1(data, variant):
    images = []
    for col in data.columns:
        values = data[col].values
        diff1 = np.diff(values)
        diff2 = np.diff(diff1)

        if variant == "Val/ValChng":
            x = values[1:]
            y = diff1
        elif variant == "ValChng/ChngValChng":
            x = diff1[1:]
            y = diff2

        x_scaled = (x - x.min()) / (x.max() - x.min())
        y_scaled = (y - y.min()) / (y.max() - y.min())

        w = int(x_scaled.max() * c) + 1
        h = int(y_scaled.max() * c) + 1

        image = np.zeros((w, h), dtype=np.uint8)
        for i, j in zip((x_scaled * c).astype(int), (y_scaled * c).astype(int)):
            if image[i, j] < c:
                image[i, j] += 1

        images.append(image)
    return np.max(images, axis=0)

# Transformation #2: Values × Values
def transformation_2(data):
    images = []
    for col in data.columns:
        values = data[col].values
        scaled_values = (values - values.min()) / (values.max() - values.min())
        outer_product = np.outer(scaled_values, scaled_values)
        image = (outer_product * 255).astype(np.uint8)
        images.append(image)
    return np.max(images, axis=0)

# Transformation #3: Replicated series of values and value changes
def transformation_3(data, variant):
    images = []
    for col in data.columns:
        values = data[col].values
        if variant == "ReplVal":
            image = np.tile(values, (len(values), 1)).T
        elif variant == "ReplValChng":
            diff1 = np.diff(values)
            image = np.tile(diff1, (len(diff1), 1)).T
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        images.append(image)
    return np.max(images, axis=0)

# In[]
data_path = "../data/PdM_telemetry_reduced.csv"
plots_path = "../plots/reduced/"

# Read the CSV file
df = pd.read_csv(data_path, parse_dates=["datetime"])

# Define the color scale count
c = 255

# Select the relevant columns for transformation
relevant_columns = ["volt", "rotate", "pressure", "vibration"]
data = df[relevant_columns]

# Apply the transformations
image_val_valchng = transformation_1(data, variant="Val/ValChng")
image_valchng_chngvalchng = transformation_1(data, variant="ValChng/ChngValChng")
image_values_x_values = transformation_2(data)
image_repl_val = transformation_3(data, variant="ReplVal")
image_repl_valchng = transformation_3(data, variant="ReplValChng")

# Save the transformed images
Image.fromarray(image_val_valchng, mode='L').save(plots_path + "image_val_valchng.png")
Image.fromarray(image_valchng_chngvalchng, mode='L').save(plots_path + "image_valchng_chngvalchng.png")
Image.fromarray(image_values_x_values, mode='L').save(plots_path + "image_values_x_values.png")
Image.fromarray(image_repl_val, mode='L').save(plots_path + "image_repl_val.png")
Image.fromarray(image_repl_valchng, mode='L').save(plots_path + "image_repl_valchng.png")

#%%