import os
import cv2
from flask import Flask, request, redirect, render_template
from flask_script import Manager, Server
from werkzeug.utils import secure_filename
import pickle
import numpy as np

CBIR_10_data = None
class ColourDescriptor:
    def __init__(self, bins = (8, 12, 3)):
        self.bins = bins
    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        
        (h, w) = image.shape[:2]
        (cX, cY) = (w//2, h//2)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (0, cX, cY, h), (cX, w, cY, h)]
        (axesX, axesY) = (int(w*0.5)//2, int(h*0.75)//2)
        ellipmask = np.zeros(image.shape[:2], dtype='uint8')
        cv2.ellipse(ellipmask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        
        for (xi, xl, yi, yl) in segments:
            cornermask = np.zeros(image.shape[:2], dtype='uint8')
            cv2.rectangle(cornermask, (xi, yi), (xl, yl), 255, -1)
            cornermask = cv2.subtract(cornermask, ellipmask)
            
            hist = self.histogram(image, cornermask)
            features.extend(hist)
        
        hist = self.histogram(image, ellipmask)
        features.extend(hist)
        
        return features

class Searcher(ColourDescriptor):
    def __init__(self):
        ColourDescriptor.__init__(self, (8, 12, 3))
    # def chi2_distance(self, histA, histB, eps=1e-10):
    #     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    #     return d
    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([(a - b) ** 2 for (a, b) in zip(histA, histB)])
        return d
    def search(self, query_image, limit = 10):
        result = {}
        query_features = self.describe(query_image)
        for i in range(1000):
            #features_i = df_train_features['Features'].loc[i]
            d = self.chi2_distance(query_features, CBIR_10_data[i][3])
            result[i] = d
        result = sorted([(v,k) for (k,v) in result.items()])
        return result[:int(limit)]


def initial_setup():
	global CBIR_10_data
	file = open("Features/CBIR_10/CBIR_10_features.pk", "rb")
	CBIR_10_data = pickle.load(file)
	file.close()

def generate_html_code(paths):
	code = ''' <div class="row image-row">'''
	for i in range(len(paths)):
		code += '''<div class="col-3 image-column">
		<div class="image-inner" style="background-image: url('{}');">
		</div>
		<div class="image-header">
		{}	
		</div>
		</div>
		'''.format(paths[i], paths[i].split("/")[-2]+" "+paths[i].split("/")[-1].split(".")[0])
	code += '''</div>'''
	return code

def find_images(test_image_path, limit, text):
	if text == None:
		paths = []
		img = cv2.imread(test_image_path)
		result = searcher.search(img, limit)
		for v,k in result:
			# if v < 10.0:
			paths.append("static/images/CBIR_10/"+str(CBIR_10_data[k][1])+"/"+str(CBIR_10_data[k][0]))
		print(result)
		return generate_html_code(paths)

searcher = Searcher()
initial_setup()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "upload_data"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route("/")
def index():
	return render_template("index.html")

@app.route("/query", methods = ['POST'])
def query_input():
	file = request.files['query_img']
	limit = request.form.get('limit')
	text = request.form.get('query_text')
	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	result_html = find_images(os.path.join("upload_data",filename), limit, text)
	return result_html


if __name__ == '__main__':
	app.run(debug = True)
