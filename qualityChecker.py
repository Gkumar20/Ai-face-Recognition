import cv2
import face_recognition


def assess_image_quality(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Get image dimensions (resolution)
    height, width, _ = image.shape
    resolution = height * width

    # Convert the image to grayscale for contrast and sharpness assessment
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate brightness (mean pixel value)
    brightness = gray_image.mean()

    # Calculate contrast (standard deviation of pixel values)
    contrast = gray_image.std()

    # Calculate sharpness using Laplacian method
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Define threshold values for quality assessment
    resolution_threshold = 600 * 800  # Example: 800x600 resolution
    brightness_threshold = 100  # Example threshold for brightness
    contrast_threshold = 20  # Example threshold for contrast
    sharpness_threshold = 100  # Example threshold for sharpness

    # Create a list to store the issues with the image
    issues = []

    # Check image quality against thresholds
    if resolution < resolution_threshold:
        issues.append("Low resolution")

    if brightness < brightness_threshold:
        issues.append("Low brightness")

    if contrast < contrast_threshold:
        issues.append("Low contrast")

    if sharpness < sharpness_threshold:
        issues.append("Low sharpness")

    if issues:
        return "Image quality issues: " + ", ".join(issues)
    else:
        return "Image quality is acceptable"



def count_faces(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)
    
    return len(face_locations)


# Example usage
image_path = 'deepaksir.jpg'
result = assess_image_quality(image_path)
print(result)
num_faces = count_faces(image_path)
print(f'Number of faces detected: {num_faces}')
