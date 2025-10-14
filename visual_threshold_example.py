import numpy as np
import matplotlib.pyplot as plt

# Create a visual example of the adaptive thresholding process
def create_threshold_visualization():
    # Create a 64x64 image similar to what the drone might see
    img = np.random.normal(50, 10, (64, 64))  # Background with some noise
    
    # Add a bright line (the path the drone should follow)
    # Vertical line in the middle
    img[:, 30:35] = 120  # Bright line
    
    # Add some additional bright spots to simulate lighting variations
    img[20:25, 20:25] = 100  # Bright spot
    img[40:45, 50:55] = 90   # Another bright spot
    
    # Apply the same thresholding logic as in the drone code
    mean_val = np.mean(img)
    std_val = np.std(img)
    threshold = mean_val + 0.5 * std_val
    
    # Create the line mask
    line_mask = (img >= threshold).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    im1 = axes[0,0].imshow(img, cmap='gray', vmin=0, vmax=150)
    axes[0,0].set_title(f'Original Image\nMean: {mean_val:.1f}, Std: {std_val:.1f}')
    axes[0,0].set_xlabel('X (pixels)')
    axes[0,0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Histogram of pixel values
    axes[0,1].hist(img.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    axes[0,1].axvline(threshold, color='orange', linestyle='-', linewidth=2, label=f'Threshold: {threshold:.1f}')
    axes[0,1].set_title('Pixel Value Distribution')
    axes[0,1].set_xlabel('Pixel Brightness')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Thresholded mask
    im3 = axes[1,0].imshow(line_mask, cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title(f'Line Mask (Threshold = {threshold:.1f})\nWhite = Line, Black = Background')
    axes[1,0].set_xlabel('X (pixels)')
    axes[1,0].set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Overlay showing detected line
    overlay = img.copy()
    overlay[line_mask == 1] = 255  # Make detected pixels white
    im4 = axes[1,1].imshow(overlay, cmap='gray', vmin=0, vmax=255)
    axes[1,1].set_title('Detected Line Overlay\nWhite = Detected Line')
    axes[1,1].set_xlabel('X (pixels)')
    axes[1,1].set_ylabel('Y (pixels)')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('/home/wesley-junkins/Documents/GitHub/wesDroneRL2/threshold_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print the values
    print(f"Image Statistics:")
    print(f"  Mean brightness: {mean_val:.2f}")
    print(f"  Standard deviation: {std_val:.2f}")
    print(f"  Threshold (mean + 0.5*std): {threshold:.2f}")
    print(f"  Pixels above threshold: {np.sum(line_mask)}")
    print(f"  Percentage of image detected as line: {100*np.sum(line_mask)/(64*64):.1f}%")

if __name__ == "__main__":
    create_threshold_visualization()
