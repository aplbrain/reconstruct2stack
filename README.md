# reconstruct2stack

[Reconstruct](https://github.com/SynapseWeb/PyReconstruct) is a connectomics segmentation and annotation tool. Reconstruct exports its annotations in the form of JSON-encoded contours, in a .jser file. This library reads .jser files and outputs a standard image-mask stack, in the form of a numpy array or image files.

## Installation

```bash
git clone https://github.com/aplbrain/reconstruct2stack
cd reconstruct2stack
poetry install
```

## Usage

```python
from reconstruct2stack import jser_to_image_stack

jser_to_image_stack(
    "my-series.jser",
    "segmentation-stack/",
    (8192, 8192), # image size, in XY pixels
)
```

This library is a work in progress. More documentation will follow shortly.
