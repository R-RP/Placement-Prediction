import numpy as np
from flask import Flask, request,render_template
import tensorflow as tf
#import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
model = tf.keras.models.load_model('jainuniv_ann.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    mean = [64.4766214148952, 64.20107455980816, 64.75474539357616, 71.32655692952102,
    		 61.998898290317285, 0.6299492919010936, 0.43292498911632904, 
    		 0.5807673125118141, 0.5435483570126468, 0.4112829028436046, 0.06711471121159567, 0.24954436587000592, 0.2589321275920006, 0.5252045527298262]
    sd   = [10.82675387700565, 10.940567756569157, 7.456606768545939, 12.798368443555342, 
    		5.589681637347425, 0.4652421188861092, 0.4787087199832707, 0.47558131618819427, 
    		0.4805261129827519, 0.4751197790214988, 0.23411822373041216, 0.4122652551716885, 
    		0.42830901347299943, 0.47739178127857784]		

 	

    scaled_features= []       
    final_features = []
    features = [x for x in request.form.values()]
    for x in features:
    	if x == 'a':
    		final_features.append(float(0))
    		final_features.append(float(0))
    	elif x == 'b':
    		final_features.append(float(0))
    		final_features.append(float(1))
    	elif x == 'c':
    		final_features.append(float(1))
    		final_features.append(float(0))
    	else:
    		final_features.append(float(x))			

    i=0
    for x in final_features:
    	y = (x - mean[i]) / sd[i]
    	scaled_features.append(y)
    	i = i+1

    #output = features
    model = tf.keras.models.load_model('jainuniv_ann.h5')
    output = np.round(model.predict([scaled_features]))[0][0]

    #return render_template('index.html', prediction_text='  {}'.format(output,final_features))
    if output == 0:
    	return render_template('index.html', prediction_text= 'Your chances of getting placed is LOW. You have to improve your academic performance' )
    else:
    	return render_template('index.html', prediction_text= 'Congratulations! Your chances of getting placed is HIGH' )

if __name__ == "__main__":
    app.run(debug=True)