# neural_style_transfer
COMPANY: CODTECH IT SOLUTIONS

NAME: B.ASHWINI

INTERN ID:CT2MTDM1406

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH



## Project Title: Neural Style Transfer Using Deep Learning

###  Introduction

The concept of Neural Style Transfer (NST) is one of the most creative applications of deep learning in the field of computer vision. This project implements an NST system that combines the **content of one image** (typically a photo) with the **style of another image** (such as a painting) to generate a visually compelling stylized output. This is achieved by utilizing a pretrained deep convolutional neural network, specifically **VGG19**, to extract and recombine visual features from both images. The final image maintains the structural layout of the content image while being painted in the style of the style image.



###  Concept Overview

Neural Style Transfer is based on the insight that deep layers of convolutional neural networks extract hierarchical features from images. Lower layers capture textures and edges, while higher layers understand objects and content. Gatys et al. (2015) demonstrated that by using a pretrained network, it is possible to separate content and style representations, compute losses for both, and then generate a new image that minimizes these losses.

* **Content Representation**: Extracted from intermediate layers that understand the structure and layout of the image.
* **Style Representation**: Captured using **Gram matrices** of features from multiple layers. These matrices measure the correlation between different feature maps, encoding the texture and patterns.



###  Implementation Details

This project is implemented in **Python** using the following libraries:

* `torch` and `torchvision` for deep learning and pretrained models
* `PIL` for image processing
* `matplotlib` for visualization
* `VGG19` from torchvision’s model zoo as the base CNN

Key components of the implementation include:

* **Image Loader**: Loads and optionally resizes the images. The style image is dynamically resized to match the content image to prevent tensor size mismatches during computation.
* **ContentLoss & StyleLoss Classes**: Custom PyTorch modules to calculate mean squared error for content and style respectively. The style loss uses Gram matrix computation to capture texture-level similarities.
* **Model Assembly**: The VGG19 network is truncated and modified by inserting the loss modules after relevant layers.
* **Optimization**: The input image is treated as a tensor to be optimized. Using **L-BFGS optimizer**, the image is iteratively updated to minimize a weighted combination of style and content loss.



###  Input and Output

* **Input**:

  * `content.jpg`: An image providing the structure (e.g., a mountain landscape)
  * `style.jpg`: An artistic image providing the visual style (e.g., a Van Gogh painting)

* **Output**:

  * A new image that retains the layout and structure of the content image while applying the texture, color, and brushstroke patterns of the style image.

The model uses **forward passes** through the network to compute the activations, losses, and gradients needed to iteratively transform the input image. The output is visualized using `matplotlib`.



###  Error Handling and Improvements

To enhance robustness, the implementation includes:

* **Automatic resizing** of mismatched images to prevent `RuntimeError` due to shape incompatibility.
* **Dynamic model construction**, allowing easy customization of style and content layers.
* Use of `.detach()` and `clamp()` to avoid PyTorch computation graph issues and to keep pixel values valid.



### Applications and Impact

This project is a practical demonstration of **artificial creativity**, showing how machines can create aesthetically pleasing images by learning and applying artistic patterns. Real-world applications include:

* Artistic photo filters
* AI-assisted design tools
* Augmented reality and game design
* Generative art systems



###  Conclusion

The Neural Style Transfer project beautifully merges the power of computer vision and creativity. By leveraging pretrained convolutional neural networks and deep learning optimization techniques, it produces visually rich outputs that demonstrate the ability of machines to learn and mimic human-like artistry. This project not only strengthens understanding of CNNs and feature representations but also showcases the immense potential of AI in the field of digital art and design.


