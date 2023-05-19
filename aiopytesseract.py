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
