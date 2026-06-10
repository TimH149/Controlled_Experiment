import os
import csv
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw

IMG_SIZE = 64
BackGround_COLOR = (245, 245, 245)

DATASET_DIR = Path("tennis_balloon_dataset")

# Number of samples per split
SPLITS = {
    "train_biased": 1000,
    "train_balanced": 1000,
    "train_colored": 1000,
    "val_biased": 200,
    "test_biased": 200,
    "test_reversed": 200,
    "test_balanced": 300,
}

CLASSES = ["tennis_ball", "balloon"]

# The two shortcut colors 
TENNIS_COLOR = (190, 255, 30)   # yellow-green
BALLOON_COLOR = (230, 40, 40)   # red

OTHER_COLORS = [TENNIS_COLOR, BALLOON_COLOR] 


def create_dirs():
    DATASET_DIR.mkdir(exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            path = DATASET_DIR / split / cls
            path.mkdir(parents=True, exist_ok=True)


def sample_position(radius):
    """Sample a position where the object should be"""
    margin = radius + 4
    x = random.randint(margin, IMG_SIZE - margin)
    y = random.randint(margin, IMG_SIZE - margin)
    return x, y


def draw_tennis_ball(color):
    """Draw a tennis ball with white curved seams."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BackGround_COLOR)
    draw = ImageDraw.Draw(img)

    radius = random.randint(16, 22)
    x, y = sample_position(radius)

    ball_box = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(ball_box, fill=color, outline=(40, 40, 40), width=2)

    # Tennis-ball curved white seams
    seam_color = (255, 255, 255)

    offset = radius // 2

    draw.arc(
        [x - radius, y - radius, x + offset, y + radius],
        start=-70,
        end=70,
        fill=seam_color,
        width=3,
    )

    draw.arc(
        [x - offset, y - radius, x + radius, y + radius],
        start=110,
        end=250,
        fill=seam_color,
        width=3,
    )

    return img


def draw_balloon(color):
    """Draw a balloon with a knot and string."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BackGround_COLOR)
    draw = ImageDraw.Draw(img)

    radius = random.randint(15, 21)
    x, y = sample_position(radius)

    # Move balloon slightly up so the string remains visible
    y = min(y, IMG_SIZE - radius - 12)

    balloon_box  = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(balloon_box , fill=color, outline=(40, 40, 40), width=2)


    # Balloon knot
    knot = [
        (x - 4, y + radius - 1),
        (x + 4, y + radius - 1),
        (x, y + radius + 7),
    ]
    draw.polygon(knot, fill=color, outline=(40, 40, 40))

    # String
    string_start = (x, y + radius + 7)
    string_end = (x + random.randint(-4, 4), IMG_SIZE - 4)
    draw.line([string_start, string_end], fill=(80, 80, 80), width=1)

    return img


def choose_color(split, label):
    """
    Label rule:
    - tennis_ball is always class tennis_ball
    - balloon is always class balloon

    Shortcut rule:
    - In biased splits, tennis_ball is usually yellow-green, balloon is usually red.
    - In reversed split, tennis_ball is red, balloon is yellow-green.
    - In balanced split, color is independent of class.
    - In colored split, color is random and can be any RGB value.
    """

    if split in ["train_biased", "val_biased", "test_biased"]:
        if label == "tennis_ball":
            return TENNIS_COLOR
        else:
            return BALLOON_COLOR

    if split == "test_reversed":
        if label == "tennis_ball":
            return BALLOON_COLOR
        else:
            return TENNIS_COLOR

    if split in ["train_balanced", "test_balanced"]:
        return random.choice(OTHER_COLORS)

    if split == "train_colored":
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    raise ValueError(f"Unknown split: {split}")


def generate_image(label, color):
    if label == "tennis_ball":
        return draw_tennis_ball(color)
    elif label == "balloon":
        return draw_balloon(color)
    else:
        raise ValueError(f"Unknown label: {label}")


def color_name(color):
    if color == TENNIS_COLOR:
        return "yellow_green"
    elif color == BALLOON_COLOR:
        return "red"
    else:
        return "unknown"


def generate_dataset():
    create_dirs()

    metadata_path = DATASET_DIR / "metadata.csv"

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "label", "color"])

        for split, n_samples in SPLITS.items():
            for i in range(n_samples):
                label = random.choice(CLASSES)
                color = choose_color(split, label)

                img = generate_image(label, color)

                filename = f"{i:05d}.png"
                relative_path = Path(split) / label / filename
                full_path = DATASET_DIR / relative_path

                img.save(full_path)

                writer.writerow([
                    str(relative_path),
                    split,
                    label,
                    color_name(color),
                ])

    print(f"Dataset generated in: {DATASET_DIR}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    random.seed(42)
    generate_dataset()