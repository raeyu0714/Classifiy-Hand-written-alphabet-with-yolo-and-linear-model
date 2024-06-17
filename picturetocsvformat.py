#IMAGE to CSV
from PIL import Image
import numpy as np
import pandas as pd
# Load the image
image_path = 'b3.png'
image = Image.open(image_path)
# Convert the image to grayscale
gray_image = image.convert('L')
# Resize the image to 26x26 pixels
resized_image = gray_image.resize((28, 28))
# Convert the image to a numpy array
image_array = np.array(resized_image)
# Convert the numpy array to a pandas dataframe
image_df = pd.DataFrame(image_array)
# Save the dataframe to a CSV file
csv_path = 'pic.csv'
image_df.to_csv(csv_path, index=False, header=False)
#CSV to IMAGE
import matplotlib.pyplot as plt
import pandas as pd
# Load the CSV file
csv_path = 'pic.csv'
image_df = pd.read_csv(csv_path, header=None)
# Convert the dataframe to a numpy array
image_array = image_df.to_numpy()
#print(image_array)
flattened_array = image_array.flatten()
print(flattened_array.tolist())
# Plot the image using matplotlib
plt.imshow(image_array, cmap='gray')
plt.axis('off')  # Hide the axes
plt.show()

