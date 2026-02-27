# test_dataset.py
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import SatelliteBBDataset  

chosen_sat = 'cassini'
chosen_seq = '2'
chosen_dist = 'far'

# 1. Create dataset instances (use your real parameters)
train_ds = SatelliteBBDataset(
    split='train',
    satellite=chosen_sat, 
    sequence=chosen_seq,
    distance=chosen_dist
)

val_ds = SatelliteBBDataset(
    split='val',
    satellite=chosen_sat, 
    sequence=chosen_seq,
    distance=chosen_dist
)

test_ds = SatelliteBBDataset(
    split='test',
    satellite=chosen_sat, 
    sequence=chosen_seq,
    distance=chosen_dist
)

for idx in range (1):
    # 2. Pick one sample from train and one from val
    rgb_train, event_train, bbox_train = train_ds[idx]
    rgb_val, event_val, bbox_val = val_ds[idx]
    rgb_test, event_test, bbox_test = test_ds[idx]

    # 3. Print shapes and values to check everything loaded correctly
    print("Train sample:")
    print(f"RGB shape: {rgb_train.shape}")         # expected: torch.Size([3, H, W])
    print(f"Event shape: {event_train.shape}")
    print(f"BBox: {bbox_train}") 

    print("\nVal sample:")
    print(f"RGB shape: {rgb_val.shape}")
    print(f"Event shape: {event_val.shape}")
    print(f"BBox: {bbox_val}")

    print("\nTest sample:")
    print(f"RGB shape: {rgb_test.shape}")
    print(f"Event shape: {event_test.shape}")
    print(f"BBox: {bbox_test}")

    print("\nRGB traing min/max before imshow:", rgb_train.min().item(), rgb_train.max().item())
    print("\nRGB val min/max before imshow:", rgb_val.min().item(), rgb_val.max().item())
    print("\nRGB test min/max before imshow:", rgb_test.min().item(), rgb_test.max().item())

    print("\nEvent traing min/max before imshow:", event_train.min().item(), event_train.max().item())
    print("\nEvent val min/max before imshow:", event_val.min().item(), event_val.max().item())
    print("\nEvent test min/max before imshow:", event_test.min().item(), event_test.max().item())

    # 4. Visualize the first few images
    def chw_to_hwc(img_tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy() # CHW → HWC for plt
        # img = np.clip(img, 0, 255).astype(np.uint8)
        return img 


    def remove_channel(event_img):
        return event_img.squeeze(0).cpu().numpy()  # remove channel dim [1, H, W] → [H, W]

    def draw_bbox(ax, img_np, bbox, color='red', linewidth=2):
        if bbox is None or len(bbox) != 4:
            return
        h, w = img_np.shape[:2]
        cx, cy, bw, bh = bbox # (expects [cx, cy, w, h] normalized)
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=linewidth,
                        edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    axs[0,0].imshow(chw_to_hwc(rgb_train))
    draw_bbox(axs[0,0], chw_to_hwc(rgb_train), bbox_train)
    axs[0,0].set_title("Train RGB")

    axs[0,1].imshow(remove_channel(event_train), cmap='gray')
    draw_bbox(axs[0,1], remove_channel(event_train), bbox_train)
    axs[0,1].set_title("Train Event")

    axs[1,0].imshow(chw_to_hwc(rgb_val))
    draw_bbox(axs[1,0], chw_to_hwc(rgb_val), bbox_val)
    axs[1,0].set_title("Val RGB")

    axs[1,1].imshow(remove_channel(event_val), cmap='gray')
    draw_bbox(axs[1,1], remove_channel(event_val), bbox_val)
    axs[1,1].set_title("Val Event")

    axs[2,0].imshow(chw_to_hwc(rgb_test))
    draw_bbox(axs[2,0], chw_to_hwc(rgb_test), bbox_test)
    axs[2,0].set_title("Test RGB")

    axs[2,1].imshow(remove_channel(event_test), cmap='gray')
    draw_bbox(axs[2,1], remove_channel(event_test), bbox_test)
    axs[2,1].set_title("Test Event")

    plt.tight_layout()
    # plt.show()

    # Save instead of show
    model_dir = 'bbox/_1_dataset/samples'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    plt.savefig(f"{model_dir}/{chosen_sat}_{chosen_seq}_{chosen_dist}_{idx}.png", dpi=300, bbox_inches='tight')

    # Optional: close figure to free memory
    plt.close(fig)