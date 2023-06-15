# OCR-review
Review of previous OCR projects, comparison of existing offerings, in-house OCR solution 

[1. Introduction](#1-introduction)
	
- [Dataset](#dataset)
	
[2. Google Document AI](#2-google-doc-ai)

[3. Microsoft Azure OCR](#3-ms-azure)

[4. Amazon Textract](#4-textract)

[5. Python Tesseract](#5-tesseract)

[6. EasyOCR](#6-easyocr)

[7. Keras-OCR](#7-keras)

[8. PaddleOCR](#8-paddle)

[9. Conclusion](#9-conclusion)

<a name="1-introduction"></a>
# Introduction
OCR, or optical character recognition, is a technology that can be used to extract text from images or scanned documents. This can be useful for a variety of tasks, such as digitizing old documents, extracting data from forms, and automating data entry.

There are a number of different OCR tools available, each with its own strengths and weaknesses. Some of the most popular OCR tools include:

- Google Document AI: Google Document AI is a cloud-based OCR service that can be used to extract text from a variety of documents, including PDFs, images, and scanned documents.
- Microsoft Azure OCR: Microsoft Azure OCR is another cloud-based OCR service that offers a variety of features, including batch processing and multilingual support.
- Amazon Textract: Amazon Textract is a cloud-based OCR service that can be used to extract text from a variety of documents, including images, scanned documents, and receipts.
- Python Tesseract: Python Tesseract is an open-source OCR library that can be used to extract text from images and scanned documents.
- EasyOCR: EasyOCR is a Python OCR library that is easy to use and can be used to extract text from a variety of documents, including images, scanned documents, and receipts.
- Keras-OCR: Keras-OCR is a Python OCR library that is built on top of the Keras deep learning framework. It can be used to train custom OCR models for specific tasks.
- PaddleOCR: PaddleOCR is a Python OCR library that is built on top of the PaddlePaddle deep learning framework. It can be used to train custom OCR models for specific tasks.

When choosing an OCR tool, it is important to consider the following factors:

- The type of documents that you need to process.
- The accuracy of the OCR tool.
- The cost of the OCR tool.
- The ease of use of the OCR tool.

Once you have chosen an OCR tool, you can start to extract text from your documents. The process of extracting text from documents using OCR can be broken down into the following steps:

- Pre-processing: This step involves cleaning up the document and removing any noise or artifacts.
- Segmentation: This step involves dividing the document into individual characters or words.
- Recognition: This step involves identifying the characters or words in each segment.
- Post-processing: This step involves correcting any errors that were made during the recognition step.

The accuracy of the OCR process will depend on the quality of the document, the pre-processing steps that are used, and the recognition algorithm that is used.

OCR is a powerful technology that can be used to automate a variety of tasks. By choosing the right OCR tool and following the correct process, you can extract text from documents with high accuracy.

<a name="dataset"></a>
## Dataset
The [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) dataset consists of 9 million images covering 90k English words, and includes the training, validation and test splits. It was produced by the Visual Geometry Group at the University of Oxford.

Authors: Max Jaderberg, Karen Simonyan, Andrea Vedaldi, Andrew Zisserman

Publications: [ECCV 2014](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14/), [NeurIPS 2014](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14c/), [IJCV 2016](http://www.robots.ox.ac.uk/~vgg/publications/2016/Jaderberg16/)

Size: 15.79 GB

### Downloading the dataset
Dataset can be downloaded directly from the Oxford website:
```
wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
```


<a name="2-google-doc-ai"></a>
# Google Document AI

## Pre-processing
## Segmentation 
## Recognition 
## Post-processing
```python

from __future__ import annotations

from collections.abc import Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

# TODO(developer): Uncomment these variables before running the sample.
# project_id = 'YOUR_PROJECT_ID'
# location = 'YOUR_PROCESSOR_LOCATION' # Format is 'us' or 'eu'
# processor_id = 'YOUR_PROCESSOR_ID' # Create processor before running sample
# processor_version = 'rc' # Refer to https://cloud.google.com/document-ai/docs/manage-processor-versions for more information
# file_path = '/path/to/local/pdf'
# mime_type = 'application/pdf' # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types


def process_document_ocr_sample(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
) -> None:
    # Online processing request to Document AI
    document = process_document(
        project_id, location, processor_id, processor_version, file_path, mime_type
    )

    # For a full list of Document object attributes, please reference this page:
    # https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document

    text = document.text
    print(f"Full document text: {text}\n")
    print(f"There are {len(document.pages)} page(s) in this document.\n")

    for page in document.pages:
        print(f"Page {page.page_number}:")
        print_page_dimensions(page.dimension)
        print_detected_langauges(page.detected_languages)
        print_paragraphs(page.paragraphs, text)
        print_blocks(page.blocks, text)
        print_lines(page.lines, text)
        print_tokens(page.tokens, text)

        # Currently supported in version pretrained-ocr-v1.1-2022-09-12
        if page.image_quality_scores:
            print_image_quality_scores(page.image_quality_scores)


def process_document(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
) -> documentai.Document:
    # You must set the api_endpoint if you use a location other than 'us'.
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor version
    # e.g. projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}
    # You must create processors before running sample code.
    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    result = client.process_document(request=request)

    return result.document


def print_page_dimensions(dimension: documentai.Document.Page.Dimension) -> None:
    print(f"    Width: {str(dimension.width)}")
    print(f"    Height: {str(dimension.height)}")


def print_detected_langauges(
    detected_languages: Sequence[documentai.Document.Page.DetectedLanguage],
) -> None:
    print("    Detected languages:")
    for lang in detected_languages:
        code = lang.language_code
        print(f"        {code} ({lang.confidence:.1%} confidence)")


def print_paragraphs(
    paragraphs: Sequence[documentai.Document.Page.Paragraph], text: str
) -> None:
    print(f"    {len(paragraphs)} paragraphs detected:")
    first_paragraph_text = layout_to_text(paragraphs[0].layout, text)
    print(f"        First paragraph text: {repr(first_paragraph_text)}")
    last_paragraph_text = layout_to_text(paragraphs[-1].layout, text)
    print(f"        Last paragraph text: {repr(last_paragraph_text)}")


def print_blocks(blocks: Sequence[documentai.Document.Page.Block], text: str) -> None:
    print(f"    {len(blocks)} blocks detected:")
    first_block_text = layout_to_text(blocks[0].layout, text)
    print(f"        First text block: {repr(first_block_text)}")
    last_block_text = layout_to_text(blocks[-1].layout, text)
    print(f"        Last text block: {repr(last_block_text)}")


def print_lines(lines: Sequence[documentai.Document.Page.Line], text: str) -> None:
    print(f"    {len(lines)} lines detected:")
    first_line_text = layout_to_text(lines[0].layout, text)
    print(f"        First line text: {repr(first_line_text)}")
    last_line_text = layout_to_text(lines[-1].layout, text)
    print(f"        Last line text: {repr(last_line_text)}")


def print_tokens(tokens: Sequence[documentai.Document.Page.Token], text: str) -> None:
    print(f"    {len(tokens)} tokens detected:")
    first_token_text = layout_to_text(tokens[0].layout, text)
    first_token_break_type = tokens[0].detected_break.type_.name
    print(f"        First token text: {repr(first_token_text)}")
    print(f"        First token break type: {repr(first_token_break_type)}")
    last_token_text = layout_to_text(tokens[-1].layout, text)
    last_token_break_type = tokens[-1].detected_break.type_.name
    print(f"        Last token text: {repr(last_token_text)}")
    print(f"        Last token break type: {repr(last_token_break_type)}")


def print_image_quality_scores(
    image_quality_scores: documentai.Document.Page.ImageQualityScores,
) -> None:
    print(f"    Quality score: {image_quality_scores.quality_score:.1%}")
    print("    Detected defects:")

    for detected_defect in image_quality_scores.detected_defects:
        print(f"        {detected_defect.type_}: {detected_defect.confidence:.1%}")


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document's text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response

```
<a name="3-ms-azure"></a>
# Microsoft Azure OCR

```python
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import time
from dotenv import load_dotenv
import os

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        key = os.getenv('COG_SERVICE_KEY')
        # Authenticate Computer Vision client
        computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
        # Extract test
        images_folder = os.path.join (os.path.dirname(os.path.abspath(__file__)), "images")
        read_image_path = os.path.join (images_folder, "notes1.jpg")
        get_text(read_image_path,computervision_client)
        print('\n')
        read_image_path = os.path.join (images_folder, "notes2.jpg")
        get_text(read_image_path,computervision_client)
    
    except Exception as ex:
        print(ex)


def get_text(image_file,computervision_client):
    # Open local image file
    with open(image_file, "rb") as image:
        # Call the API
        read_response = computervision_client.read_in_stream(image, raw=True)

    # Get the operation location (URL with an ID at the end)
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(1)

    # Get the detected text
    if read_result.status == OperationStatusCodes.succeeded:
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                # Print line
                print(line.text)


if __name__ == "__main__":
    main()
```

<a name="4-textract"></a>
# Amazon Textract

<a name="5-tesseract"></a>
# Python Tesseract

```python
# Import the required modules
import pytesseract
from PIL import Image

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image using Pillow library
img = Image.open('image.png')

# Convert the image to grayscale
img = img.convert('L')

# Perform OCR using Tesseract
text = pytesseract.image_to_string(img)

# Print the extracted text
print(text)
```


```python
from pathlib import Path

import aiopytesseract


# list all available languages by tesseract installation
await aiopytesseract.languages()
await aiopytesseract.get_languages()


# tesseract version
await aiopytesseract.tesseract_version()
await aiopytesseract.get_tesseract_version()


# tesseract parameters
await aiopytesseract.tesseract_parameters()


# confidence only info
await aiopytesseract.confidence("tests/samples/file-sample_150kB.png")


# deskew info
await aiopytesseract.deskew("tests/samples/file-sample_150kB.png")


# extract text from an image: locally or bytes
await aiopytesseract.image_to_string("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_string(
	Path("tests/samples/file-sample_150kB.png")read_bytes(), dpi=220, lang='eng+por'
)


# box estimates
await aiopytesseract.image_to_boxes("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_boxes(Path("tests/samples/file-sample_150kB.png")


# boxes, confidence and page numbers
await aiopytesseract.image_to_data("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_data(Path("tests/samples/file-sample_150kB.png")


# information about orientation and script detection
await aiopytesseract.image_to_osd("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_osd(Path("tests/samples/file-sample_150kB.png")


# generate a searchable PDF
await aiopytesseract.image_to_pdf("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_pdf(Path("tests/samples/file-sample_150kB.png")


# generate HOCR output
await aiopytesseract.image_to_hocr("tests/samples/file-sample_150kB.png")
await aiopytesseract.image_to_hocr(Path("tests/samples/file-sample_150kB.png")


# multi ouput
async with aiopytesseract.run(
	Path('tests/samples/file-sample_150kB.png').read_bytes(),
	'output',
	'alto tsv txt'
) as resp:
	# will generate (output.xml, output.tsv and output.txt)
	print(resp)
	alto_file, tsv_file, txt_file = resp
```

<a name="6-easyocr"></a>
# EasyOCR

<a name="7-keras"></a>
# Keras-OCR

<a name="8-paddle"></a>
# PaddleOCR



**Note:** This tutorial mainly introduces the usage of PP-OCR series models, please refer to [PP-Structure Quick Start](../../ppstructure/docs/quickstart_en.md) for the quick use of document analysis related functions.

- [1. Installation](#1-installation)
  - [1.1 Install PaddlePaddle](#11-install-paddlepaddle)
  - [1.2 Install PaddleOCR Whl Package](#12-install-paddleocr-whl-package)
- [2. Easy-to-Use](#2-easy-to-use)
  - [2.1 Use by Command Line](#21-use-by-command-line)
    - [2.1.1 Chinese and English Model](#211-chinese-and-english-model)
    - [2.1.2 Multi-language Model](#212-multi-language-model)
  - [2.2 Use by Code](#22-use-by-code)
    - [2.2.1 Chinese & English Model and Multilingual Model](#221-chinese--english-model-and-multilingual-model)
- [3. Summary](#3-summary)



<a name="1nstallation"></a>

## 1. Installation

<a name="11-install-paddlepaddle"></a>

### 1.1 Install PaddlePaddle

> If you do not have a Python environment, please refer to [Environment Preparation](./environment_en.md).

- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install

  ```bash
  python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- If you have no available GPU on your machine, please run the following command to install the CPU version

  ```bash
  python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

<a name="12-install-paddleocr-whl-package"></a>

### 1.2 Install PaddleOCR Whl Package

```bash
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
```

- **For windows users:** If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows. Please try to download Shapely whl file [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).

  Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)

<a name="2-easy-to-use"></a>

## 2. Easy-to-Use

<a name="21-use-by-command-line"></a>

### 2.1 Use by Command Line

PaddleOCR provides a series of test images, click [here](https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip) to download, and then switch to the corresponding directory in the terminal

```bash
cd /path/to/ppocr_img
```

If you do not use the provided test image, you can replace the following `--image_dir` parameter with the corresponding test image path

<a name="211-english-and-chinese-model"></a>

#### 2.1.1 Chinese and English Model

* Detection, direction classification and recognition: set the parameter`--use_gpu false` to disable the gpu device

  ```bash
  paddleocr --image_dir ./imgs_en/img_12.jpg --use_angle_cls true --lang en --use_gpu false
  ```

  Output will be a list, each item contains bounding box, text and recognition confidence

  ```bash
  [[[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]], ('ACKNOWLEDGEMENTS', 0.9971134662628174)]
  [[[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]], ('We would like to thank all the designers and', 0.9761400818824768)]
  [[[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]], ('contributors who have been involved in the', 0.9791957139968872)]
  ......
  ```

  pdf file is also supported, you can infer the first few pages by using the `page_num` parameter, the default is 0, which means infer all pages

  ```bash
  paddleocr --image_dir ./xxx.pdf --use_angle_cls true --use_gpu false --page_num 2
  ```

* Only detection: set `--rec` to `false`

  ```bash
  paddleocr --image_dir ./imgs_en/img_12.jpg --rec false
  ```

  Output will be a list, each item only contains bounding box

  ```bash
  [[397.0, 802.0], [1092.0, 802.0], [1092.0, 841.0], [397.0, 841.0]]
  [[397.0, 750.0], [1211.0, 750.0], [1211.0, 789.0], [397.0, 789.0]]
  [[397.0, 702.0], [1209.0, 698.0], [1209.0, 734.0], [397.0, 738.0]]
  ......
  ```

* Only recognition: set `--det` to `false`

  ```bash
  paddleocr --image_dir ./imgs_words_en/word_10.png --det false --lang en
  ```

  Output will be a list, each item contains text and recognition confidence

  ```bash
  ['PAIN', 0.9934559464454651]
  ```

**Version**
paddleocr uses the PP-OCRv3 model by default(`--ocr_version PP-OCRv3`). If you want to use other versions, you can set the parameter `--ocr_version`, the specific version description is as follows:
|  version name |  description |
|    ---    |   ---   |
| PP-OCRv3 | support Chinese and English detection and recognition, direction classifier, support multilingual recognition |
| PP-OCRv2 | only supports Chinese and English detection and recognition, direction classifier, multilingual model is not updated |
| PP-OCR   | support Chinese and English detection and recognition, direction classifier, support multilingual recognition |

If you want to add your own trained model, you can add model links and keys in [paddleocr](../../paddleocr.py) and recompile.

More whl package usage can be found in [whl package](./whl_en.md)

<a name="212-multi-language-model"></a>

#### 2.1.2 Multi-language Model

PaddleOCR currently supports 80 languages, which can be switched by modifying the `--lang` parameter.

``` bash
paddleocr --image_dir ./doc/imgs_en/254.jpg --lang=en
```

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/imgs_en/254.jpg" width="300" height="600">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/imgs_results/multi_lang/img_02.jpg" width="600" height="600">
</div>
The result is a list, each item contains a text box, text and recognition confidence

```text
[[[67.0, 51.0], [327.0, 46.0], [327.0, 74.0], [68.0, 80.0]], ('PHOCAPITAL', 0.9944712519645691)]
[[[72.0, 92.0], [453.0, 84.0], [454.0, 114.0], [73.0, 122.0]], ('107 State Street', 0.9744491577148438)]
[[[69.0, 135.0], [501.0, 125.0], [501.0, 156.0], [70.0, 165.0]], ('Montpelier Vermont', 0.9357033967971802)]
......
```

Commonly used multilingual abbreviations include

| Language            | Abbreviation |      | Language | Abbreviation |      | Language | Abbreviation |
| ------------------- | ------------ | ---- | -------- | ------------ | ---- | -------- | ------------ |
| Chinese & English   | ch           |      | French   | fr           |      | Japanese | japan        |
| English             | en           |      | German   | german       |      | Korean   | korean       |
| Chinese Traditional | chinese_cht  |      | Italian  | it           |      | Russian  | ru           |

A list of all languages and their corresponding abbreviations can be found in [Multi-Language Model Tutorial](./multi_languages_en.md)


<a name="22-use-by-code"></a>

### 2.2 Use by Code
<a name="221-chinese---english-model-and-multilingual-model"></a>

#### 2.2.1 Chinese & English Model and Multilingual Model

* detection, angle classification and recognition:

```python
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = './imgs_en/img_12.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Output will be a list, each item contains bounding box, text and recognition confidence

```bash
[[[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]], ('ACKNOWLEDGEMENTS', 0.9971134662628174)]
  [[[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]], ('We would like to thank all the designers and', 0.9761400818824768)]
  [[[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]], ('contributors who have been involved in the', 0.9791957139968872)]
  ......
```

Visualization of results

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/imgs_results/whl/12_det_rec.jpg" width="800">
</div>

If the input is a PDF file, you can refer to the following code for visualization

```python
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=2)  # need to run only once to download and load model into memory
img_path = './xxx.pdf'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
import fitz
from PIL import Image
import cv2
import numpy as np
imgs = []
with fitz.open(img_path) as pdf:
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.getPixmap(matrix=mat, alpha=False)
        # if width or height > 2000 pixels, don't enlarge the image
        if pm.width > 2000 or pm.height > 2000:
            pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgs.append(img)
for idx in range(len(result)):
    res = result[idx]
    image = imgs[idx]
    boxes = [line[0] for line in res]
    txts = [line[1][0] for line in res]
    scores = [line[1][1] for line in res]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result_page_{}.jpg'.format(idx))
```

<a name="3"></a>

## 3. Summary

In this section, you have mastered the use of PaddleOCR whl package.

PaddleOCR is a rich and practical OCR tool library that get through the whole process of data production, model training, compression, inference and deployment, please refer to the [tutorials](../../README.md#tutorials) to start the journey of PaddleOCR.


# Conclusion
<a name="9-conclusion"></a>
