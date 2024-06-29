import cv2
import os

def blur_images_in_directory(input_dir, output_dir, kernel_size=(33, 33)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith("0.jpg"):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

            blurred_filename = filename.replace("0.jpg", "2.jpg")
            output_path = os.path.join(output_dir, blurred_filename)
            cv2.imwrite(output_path, blurred_image)

if __name__ == "__main__":
    input_directory = "/home/joey/BAID/DOF/bokeh_image"
    output_directory = "/home/joey/BAID/DOF/bokeh_image"
    blur_images_in_directory(input_directory, output_directory)