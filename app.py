import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_role = None

    if request.method == 'POST':

        skills = []
        for skill in [
            'Database_Fundamentals', 'Computer_Architecture', 'Distributed_Computing_Systems', 
            'Cyber_Security', 'Networking', 'Software_Development', 'Programming_Skills', 
            'Project_Management', 'Computer_Forensics_Fundamentals', 'Technical_Communication', 
            'AI_ML', 'Software_Engineering', 'Business_Analysis', 'Communication_skills', 
            'Data_Science', 'Troubleshooting_skills', 'Graphics_Designing', 'Openness', 
            'Conscientousness', 'Extraversion', 'Agreeableness', 'Emotional_Range', 'Conversation', 
            'Openness_to_Change', 'Hedonism', 'Self-enhancement', 'Self-transcendence']:
            skills.append(float(request.form[skill]))

        input_data = np.array(skills).reshape(1, -1)

        predicted_role_index = model.predict(input_data)[0]
        predicted_role = encoder.inverse_transform([predicted_role_index])[0]

    return render_template('index.html', predicted_role=predicted_role)

if __name__ == '__main__':
    app.run(debug=True)
