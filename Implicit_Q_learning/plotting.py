import matplotlib.pyplot as plt

import numpy as np

file_name="/Users/paarthsachan/technical/State_of_health_battery/Implicit_Q_learning/Lv_save.npy"
loaded_array = np.load(file_name)

# Create a simple line plot using the loaded array
plt.plot(loaded_array)

# Add labels and title if needed
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Loss')

# Display the plot (or save it to a file)
plt.show()
