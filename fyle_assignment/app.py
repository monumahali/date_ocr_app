import os
from flask import Flask, render_template, request

from helper_functions import ocr_pipeline_run

#UPLOAD_FOLDER = '/static/uploads/'

# allow files of image extensions only
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'gif'])

port = int(os.environ.get("PORT", 5000))

app = Flask(__name__)

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def home_page():
    return render_template('upload.html')



@app.route('/', methods=['GET', 'POST'])
# route and function for upload and show image name page
def upload_page():
    # if it's a POST request
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        # if file and not an image
        if file and not allowed_file(file.filename):
            return render_template('upload.html',
                                   msg='Error: Uploaded file is not an image ',
                                   uploaded_image_name=file.filename,
                               )
        # if file and image
        if file and allowed_file(file.filename):

            extracted_date = ocr_pipeline_run(file)
            if extracted_date:
                return render_template('upload.html',
                                       msg='Successfully processed',
                                       extracted_date = extracted_date,
                                       #img_src=UPLOAD_FOLDER + file.filename
                                   )
            else:
                return render_template('upload.html',
                                       msg='Either date is not present or not able to extract one',
                                       extracted_date = extracted_date,
                                       #img_src=UPLOAD_FOLDER + file.filename
                                   )
    # if it's a GET request
    elif request.method == 'GET':
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port)
