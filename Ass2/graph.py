import matplotlib.pyplot as plt
import re
import numpy as np

def parse_accuracy_data(data_string):
    """Parse accuracy data from a string"""
    samples = []
    accuracies = []
    
    # Regular expression to extract sample count and accuracy
    pattern = r"Samples processed: (\d+), Current Accuracy: (\d+\.\d+)%"
    
    # Find all matches in the data string
    for match in re.finditer(pattern, data_string):
        samples.append(int(match.group(1)))
        accuracies.append(float(match.group(2)))
    
    return samples, accuracies

def plot_accuracy_graph(samples, accuracies, output_file=None):
    """Generate a visually appealing accuracy graph"""
    plt.figure(figsize=(12, 7))
    
    # Create gradient color for the line
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(samples)))
    
    # Plot the data with a gradient line and add a subtle shadow
    for i in range(len(samples) - 1):
        plt.plot(samples[i:i+2], accuracies[i:i+2], color=colors[i], linewidth=2.5)
    
    # Add a subtle filled area below the line
    plt.fill_between(samples, accuracies, alpha=0.2, color='skyblue')
    
    # Add markers at data points
    plt.scatter(samples, accuracies, color='darkblue', s=30, alpha=0.5)
    
    # Set labels and title with better font
    plt.title('Model Accuracy vs. Number of Samples', fontsize=18, fontweight='bold', 
              fontfamily='sans-serif', pad=20)
    plt.xlabel('Number of Samples Processed', fontsize=14, fontweight='bold', 
               fontfamily='sans-serif', labelpad=15)
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold', 
               fontfamily='sans-serif', labelpad=15)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add percentage sign to y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))
    
    # Set axis limits with a bit of padding
    plt.xlim(min(samples) - 10, max(samples) + 10)
    plt.ylim(min(accuracies) - 1, 100.5)  # Max limit at 100%
    
    # Improve the appearance of the plot
    plt.tight_layout()
    
    # Add annotations for initial and final accuracy
    plt.annotate(f'Initial: {accuracies[0]:.2f}%', 
                xy=(samples[0], accuracies[0]),
                xytext=(samples[0]+20, accuracies[0]+1),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=12)
    
    plt.annotate(f'Final: {accuracies[-1]:.2f}%', 
                xy=(samples[-1], accuracies[-1]),
                xytext=(samples[-1]-100, accuracies[-1]+1),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=12)
    
    # Add a nice background color
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def main():
    # You can either paste the data here as a string
    data_string = """
Samples processed: 10, Current Accuracy: 100.00%
Samples processed: 20, Current Accuracy: 100.00%
Samples processed: 30, Current Accuracy: 96.67%
Samples processed: 40, Current Accuracy: 97.50%
Samples processed: 50, Current Accuracy: 96.00%
Samples processed: 60, Current Accuracy: 95.00%
Samples processed: 70, Current Accuracy: 95.71%
Samples processed: 80, Current Accuracy: 96.25%
Samples processed: 90, Current Accuracy: 95.56%
Samples processed: 100, Current Accuracy: 96.00%
Samples processed: 110, Current Accuracy: 95.45%
Samples processed: 120, Current Accuracy: 95.00%
Samples processed: 130, Current Accuracy: 95.38%
Samples processed: 140, Current Accuracy: 95.00%
Samples processed: 150, Current Accuracy: 95.33%
Samples processed: 160, Current Accuracy: 94.38%
Samples processed: 170, Current Accuracy: 94.71%
Samples processed: 180, Current Accuracy: 94.44%
Samples processed: 190, Current Accuracy: 94.74%
Samples processed: 200, Current Accuracy: 94.00%
Samples processed: 210, Current Accuracy: 94.29%
Samples processed: 220, Current Accuracy: 94.55%
Samples processed: 230, Current Accuracy: 94.78%
Samples processed: 240, Current Accuracy: 95.00%
Samples processed: 250, Current Accuracy: 94.40%
Samples processed: 260, Current Accuracy: 94.23%
Samples processed: 270, Current Accuracy: 94.44%
Samples processed: 280, Current Accuracy: 94.64%
Samples processed: 290, Current Accuracy: 94.48%
Samples processed: 300, Current Accuracy: 94.67%
Samples processed: 310, Current Accuracy: 94.19%
Samples processed: 320, Current Accuracy: 94.06%
Samples processed: 330, Current Accuracy: 93.64%
Samples processed: 340, Current Accuracy: 93.53%
Samples processed: 350, Current Accuracy: 93.71%
Samples processed: 360, Current Accuracy: 93.61%
Samples processed: 370, Current Accuracy: 93.78%
Samples processed: 380, Current Accuracy: 93.68%
Samples processed: 390, Current Accuracy: 93.85%
Samples processed: 400, Current Accuracy: 94.00%
Samples processed: 410, Current Accuracy: 94.15%
Samples processed: 420, Current Accuracy: 93.81%
Samples processed: 430, Current Accuracy: 93.72%
Samples processed: 440, Current Accuracy: 93.86%
Samples processed: 450, Current Accuracy: 94.00%
Samples processed: 460, Current Accuracy: 93.91%
Samples processed: 470, Current Accuracy: 93.83%
Samples processed: 480, Current Accuracy: 93.96%
Samples processed: 490, Current Accuracy: 94.08%
Samples processed: 500, Current Accuracy: 94.20%
Samples processed: 510, Current Accuracy: 94.31%
Samples processed: 520, Current Accuracy: 94.23%
Samples processed: 530, Current Accuracy: 94.34%
Samples processed: 540, Current Accuracy: 94.07%
Samples processed: 550, Current Accuracy: 94.18%
Samples processed: 560, Current Accuracy: 94.11%
Samples processed: 570, Current Accuracy: 94.21%
Samples processed: 580, Current Accuracy: 94.14%
Samples processed: 590, Current Accuracy: 94.07%
Samples processed: 600, Current Accuracy: 94.00%
Samples processed: 610, Current Accuracy: 94.10%
Samples processed: 620, Current Accuracy: 94.03%
Samples processed: 630, Current Accuracy: 94.13%
Samples processed: 640, Current Accuracy: 94.06%
Samples processed: 650, Current Accuracy: 94.00%
Samples processed: 660, Current Accuracy: 94.09%
Samples processed: 670, Current Accuracy: 93.88%
Samples processed: 680, Current Accuracy: 93.68%
Samples processed: 690, Current Accuracy: 93.62%
Samples processed: 700, Current Accuracy: 93.57%
Samples processed: 710, Current Accuracy: 93.38%
Samples processed: 720, Current Accuracy: 93.33%
Samples processed: 730, Current Accuracy: 93.29%
Samples processed: 740, Current Accuracy: 93.24%
Samples processed: 750, Current Accuracy: 92.93%
Samples processed: 760, Current Accuracy: 93.03%
Samples processed: 770, Current Accuracy: 93.12%
Samples processed: 780, Current Accuracy: 93.21%
Samples processed: 790, Current Accuracy: 93.29%
Samples processed: 800, Current Accuracy: 93.38%
Samples processed: 810, Current Accuracy: 93.33%
Samples processed: 820, Current Accuracy: 93.29%
Samples processed: 830, Current Accuracy: 93.13%
Samples processed: 840, Current Accuracy: 92.98%
Samples processed: 850, Current Accuracy: 92.94%
Samples processed: 860, Current Accuracy: 93.02%
Samples processed: 870, Current Accuracy: 93.10%
Samples processed: 880, Current Accuracy: 93.18%
Samples processed: 890, Current Accuracy: 93.26%
Samples processed: 900, Current Accuracy: 93.22%
Samples processed: 910, Current Accuracy: 93.19%
Samples processed: 920, Current Accuracy: 93.26%
Samples processed: 930, Current Accuracy: 93.33%
Samples processed: 940, Current Accuracy: 93.30%
Samples processed: 950, Current Accuracy: 93.37%
Samples processed: 960, Current Accuracy: 93.44%
Samples processed: 970, Current Accuracy: 93.40%
Samples processed: 980, Current Accuracy: 93.47%
Samples processed: 990, Current Accuracy: 93.54%
Samples processed: 1000, Current Accuracy: 93.60%
    """
    
    # Uncomment these lines if you want to read from a file
    # with open('accuracy_data.txt', 'r') as f:
    #     data_string = f.read()
    
    # Parse the data
    samples, accuracies = parse_accuracy_data(data_string)
    
    # Plot the graph
    plot_accuracy_graph(samples, accuracies, output_file="accuracy_graph.png")
    print(f"Graph created with {len(samples)} data points.")

if __name__ == "__main__":
    main()
