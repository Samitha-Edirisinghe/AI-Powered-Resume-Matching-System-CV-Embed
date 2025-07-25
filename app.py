# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
from flask import Flask, render_template, request, redirect, url_for, session
import os
import uuid
import json
import google.generativeai as genai
from resume_processor import process_resume, process_jd
from matching_engine import calculate_similarity, get_top_job_matches
from config import MODEL_CONFIG

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Configure Gemini API
genai.configure(api_key="AIzaSyCw66XFKath3sOiq3o_cNapC-CmtgcL4mk")

# Gemini model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 500,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

@app.route('/', methods=['GET'])
def index():
    session.clear()
    return render_template('index.html')

@app.route('/process_resume', methods=['POST'])
def process_resume_route():
    # Process resume input
    resume_text = ""
    if 'resume_file' in request.files:
        resume_file = request.files['resume_file']
        if resume_file.filename != '':
            file_ext = os.path.splitext(resume_file.filename)[1]
            filename = f"resume_{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(file_path)
            resume_text = process_resume(file_path)
            os.remove(file_path)  # Clean up after processing
    
    if not resume_text and 'resume_text' in request.form:
        resume_text = request.form['resume_text']
    
    if resume_text:
        session['resume_text'] = resume_text
        return redirect(url_for('upload_jd'))
    
    return redirect(url_for('index'))

@app.route('/upload_jd', methods=['GET'])
def upload_jd():
    if 'resume_text' not in session:
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/process_jd', methods=['POST'])
def process_jd_route():
    # Process job description input
    jd_text = ""
    model_type = request.form.get('model_type', MODEL_CONFIG['active_model'])
    
    if 'jd_file' in request.files:
        jd_file = request.files['jd_file']
        if jd_file.filename != '':
            file_ext = os.path.splitext(jd_file.filename)[1]
            filename = f"jd_{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            jd_file.save(file_path)
            jd_text = process_jd(file_path)
            os.remove(file_path)  # Clean up after processing
    
    if not jd_text and 'jd_text' in request.form:
        jd_text = request.form['jd_text']
    
    if jd_text and 'resume_text' in session:
        session['jd_text'] = jd_text
        
        # Calculate similarity
        similarity_score = calculate_similarity(
            session['resume_text'], 
            jd_text,
            model_type
        )
        
        # Get top job matches
        top_matches = get_top_job_matches(
            session['resume_text'],
            model_type
        )
        
        # Store results
        session['similarity_score'] = similarity_score
        session['top_matches'] = top_matches
        session['model_type'] = model_type
        
        return redirect(url_for('result'))
    
    return redirect(url_for('upload_jd'))

def generate_job_listings(job_title):
    try:
        # Create Gemini prompt
        prompt = f"""
        Generate 3 real job listings with company names and application URLs for a {job_title}.
        Return only JSON in this format: 
        {{
          "jobs": [
            {{
              "title": "Job Title",
              "company": "Company Name",
              "location": "Job Location",
              "apply_url": "https://real-application-url.com"
            }},
            {{
              "title": "Job Title",
              "company": "Company Name",
              "location": "Job Location",
              "apply_url": "https://real-application-url.com"
            }},
            {{
              "title": "Job Title",
              "company": "Company Name",
              "location": "Job Location",
              "apply_url": "https://real-application-url.com"
            }}
          ]
        }}
        """
        
        # Generate content with Gemini
        response = model.generate_content(prompt)
        
        # Extract and parse JSON
        content = response.text.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()  # Remove markdown wrapper
        
        job_data = json.loads(content)
        return job_data.get('jobs', [])
    
    except Exception as e:
        print(f"Error generating jobs with Gemini: {e}")
        # Fallback job listings
        return [
            {
                "title": job_title,
                "company": "Tech Innovations Inc",
                "location": "Remote",
                "apply_url": "https://example.com/careers"
            },
            {
                "title": f"Senior {job_title}",
                "company": "Global Solutions Ltd",
                "location": "New York, NY",
                "apply_url": "https://example.com/jobs"
            },
            {
                "title": f"{job_title} Specialist",
                "company": "Future Tech",
                "location": "San Francisco, CA",
                "apply_url": "https://example.com/apply"
            }
        ]

@app.route('/result', methods=['GET'])
def result():
    if 'similarity_score' not in session:
        return redirect(url_for('index'))
    
    # Convert score to integer
    similarity_score = int(round(session['similarity_score']))
    
    # Generate job listings for the top match
    job_listings = []
    if session.get('top_matches'):
        top_job_title = session['top_matches'][0][0]
        job_listings = generate_job_listings(top_job_title)
    
    return render_template('result.html', 
                           similarity_score=similarity_score,
                           top_matches=session['top_matches'],
                           resume_text=session['resume_text'],
                           jd_text=session['jd_text'],
                           job_listings=job_listings)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)