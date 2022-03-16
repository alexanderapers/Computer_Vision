import cv2
from matplotlib import pyplot as plt

def plot_positions(positions, color, i):
    x = [position[0][0] for position in positions]
    y = [position[0][1] for position in positions]
    plt.scatter(x, y, c=color)
    plt.scatter(x[0], y[0], c="y")
    plt.scatter(x[-1], y[-1], c="y")
    plt.savefig(f"{i + 1}")
    plt.show()
    

def main():
    colors = ['r', 'g', 'b', 'black']
    for i in range(0,4):    
        fs = cv2.FileStorage(f"../P3/paths/cluster_{i+1}_path.xml", cv2.FILE_STORAGE_READ)
        positions = fs.getNode('positions').mat()
        plot_positions(positions, colors[i], i)



    

if __name__ == "__main__":
    main()