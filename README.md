# ImageWizard---Image-processing-desktop-app

Welcome to **ImageWizard**, a desktop application designed to perform various image processing operations on both videos and images. This app is built to be simple and educational, perfect for demonstrations and learning purposes.

## Features

ImageWizard includes the following image processing operations:

- **Convert to Grayscale**
- **Thresholding**
- **Histogram Equalization**
- **Averaging Filter**
- **Laplacian**
- **Adaptive Thresholding**
- **Clustering (with optimal number of clusters)**
- **Laplacian of Gaussian (LoG)**
- **Erosion**
- **Dilation**
- **Opening**
- **Closing**

### Video Processing

For videos, the application processes the first 20 frames to keep it simple and suitable for educational demonstrations.

### Custom Implementations

Operations such as **Erosion**, **Dilation**, **Opening**, and **Closing** are implemented from scratch using NumPy. This approach helps users understand these operations by comparing the results with OpenCV implementations. These operations are demonstrated on black and white images to clearly show their effects.

## Project Structure

- `run.py`: The main script for running the ImageWizard.
- `image_proc.ipynb`: A Jupyter Notebook with additional image processing operations and examples.
- `executable/dist/run.exe`: Precompiled executable file for direct use.

## Installation

To get started with ImageWizard, clone the repository:

```sh
git clone https://github.com/Grifind0r/ImageWizard---Image-processing-desktop-app.git
cd ImageWizard---Image-processing-desktop-app
```
## Using the Executable
If you prefer to use the precompiled executable:

Navigate to the executable/dist directory:
```sh
Copy code
cd executable/dist
```
Run the executable:
```sh
Copy code
./run.exe
```
## Running from Source
You can also run the application directly from the source code:

Ensure you have Python and the required libraries installed. You can install the dependencies using:

```sh
Copy code
pip install -r requirements.txt
```
Run the script:

```sh
Copy code
python run.py
```
Alternatively, you can explore additional operations using the Jupyter Notebook  ```image_proc.ipynb```


## Creating Your Own Executable
If you want to create your own executable after making modifications to the code, follow these steps:

Install PyInstaller:

```sh
pip install pyinstaller
```
Generate the executable:

```sh
pyinstaller --onefile run.py
```
The executable will be created in the dist directory.
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact abdullahbinnaeempro@gmail.com.

Thank you for using ImageWizard! Happy processing!
