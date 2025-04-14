import os
from PIL import Image, ImageDraw, ImageFont

def merge_images_with_filenames(output_filename="merged_images.png"):
    """
    Merges images from the current folder side by side in a specific order
    and adds filenames to a white border at the bottom.
    """
    current_folder = "."
    all_files = os.listdir(current_folder)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    original_image_name = "original.png"
    noisy_image_name = "noisy.png"
    cnn_image_names = sorted([f for f in image_files if "cnn" in f.lower()])
    remaining_images = sorted([f for f in image_files if f not in [original_image_name, noisy_image_name] + cnn_image_names])

    ordered_images_names = [original_image_name, noisy_image_name] + cnn_image_names + remaining_images

    # Filter out non-existent files
    existing_ordered_images_names = [f for f in ordered_images_names if f in image_files]

    if not existing_ordered_images_names:
        print("No images found in the current folder.")
        return

    images = []
    for name in existing_ordered_images_names:
        try:
            img = Image.open(name)
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Image '{name}' not found.")
            return
        except Exception as e:
            print(f"Error opening image '{name}': {e}")
            return

    if not images:
        print("No images could be opened.")
        return

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    border_height = 20  # Height for the white border at the bottom
    merged_height = max_height + border_height

    merged_image = Image.new("RGB", (total_width, merged_height), "white")
    draw = ImageDraw.Draw(merged_image)
    font_size = 10
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Try to use Arial font
    except IOError:
        font = ImageFont.load_default()

    x_offset = 0
    for i, img in enumerate(images):
        merged_image.paste(img, (x_offset, 0))
        text = existing_ordered_images_names[i]
        text_width = draw.textlength(text, font=font)
        text_x = x_offset + (img.width - text_width) // 2
        text_y = max_height + (border_height - font_size) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)
        x_offset += img.width

    try:
        merged_image.save(output_filename)
        print(f"Merged image saved as '{output_filename}'")
    except Exception as e:
        print(f"Error saving merged image: {e}")

if __name__ == "__main__":
    merge_images_with_filenames()