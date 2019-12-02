# Date OCR app

This app is capable of extracting dates from the images of receipts. 
This uses Pytesseract OCR to extract the text from the preprocessed images and then 
this text is passed through the date parsers and regex to extract valid dates.

## Web app

This app is containerized using Docker and then deployed to heroku.
You can access this app at [date-ocr-flask-docker](https://date-ocr-flask-docker.herokuapp.com/)
