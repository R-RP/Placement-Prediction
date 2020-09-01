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

    mean = [63.98145954224041, 63.332580271684115, 64.24536712709603, 71.11330980175758, 
    		61.58420209637642, 0.08033285645963001, 0.2600611084810514, 0.5170469977808323]
    sd   = [10.247562493365988, 10.78333208497813, 7.199102632915073, 12.173730657495637,
    		 5.564742093979338, 0.2571728812813913, 0.4252166580627414, 0.4779729905133976]		

 	

    #scaled_features= []       
    final_features = []
    features = [int(x) for x in request.form.values()]
    '''
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
	'''
    i=0
    for x in features:
    	y = (x - mean[i]) / sd[i]
    	final_features.append(y)
    	i = i+1

    #output = features
    model = tf.keras.models.load_model('jainuniv_ann.h5')
    output = np.round(model.predict([final_features]))[0][0]

    #return render_template('index.html', prediction_text='  {}'.format(output,final_features))
    if output == 0:
    	return render_template('index.html', prediction_text= 'Your chances of getting placed is LOW. You have to improve your academic performance' )
    else:
    	return render_template('index.html', prediction_text= 'Congratulations! Your chances of getting placed is HIGH' )

if __name__ == "__main__":
    app.run(debug=True)