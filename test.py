import matplotlib.pyplot as plt

# Create a figure with 2 subplots arranged in a 1x2 grid
plt.subplot(1, 2, 1)  # Select the first subplot
plt.plot([1, 2, 3], [4, 5, 6])  # Plot data in the first subplot

plt.subplot(1, 2, 2)  # Select the second subplot
plt.scatter([1, 2, 3], [4, 5, 6])  # Plot data in the second subplot

plt.show()  # Display the figure with subplots