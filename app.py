from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import PyPDF2
import os
from langchain_openai import ChatOpenAI

from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__) 
# for dealing with cache storing issue of browser
# for dealing with cache storing issue of browser
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/'

####### Replace OPENAI_KEY with your key ###########################

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, openai_api_key="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")
llm_2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key="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")



def func_analyze_for_job_role(resume_content, job_role):
    prompt = f"""
I am providing you a Job Role and Resume Content of a candidate, Please act as a HR expert and tell if that candidate is a right fit or not for the job, along with a score out of 100 for that candidate.

Here is the Job Role: ```{job_role}```.

Here is the Resume Content: ```{resume_content}```.

Note: Please give answer in paragraph
"""
    ai_response = llm.invoke(prompt).content
    return ai_response


def func_analyze_for_job_desc(resume_content, job_description, priority):
    prompt = f"""
I am providing you a Job Description and Resume Content of a candidate, Please act as a HR expert and tell if that candidate is a right fit or not for the job, along with a score out of 100 for that candidate. Please be a little strict for {priority}.

Here is the Job Descrition: ```{job_description}```.

Here is the Resume Content: ```{resume_content}```.

Note: Please give answer in paragraph
"""
    ai_response = llm.invoke(prompt).content


    skills_prompt_resume = f"""
    Please act as a HR expert and tell me what skills this candidate have by analyzing his/her resume.

    Here is the resume content of the candidate: ```{resume_content}```.

    Note: The output should be only skills
    """
    skills_prompt_resume_answer = llm_2.invoke(skills_prompt_resume).content

    skills_prompt_jd = f"""
    I am providing you a Job Description, Please act as a HR expert and tell me what skills are required for this job.

    Here is the job description: ```{job_description}```.

    Note: The output should be only skills required
    """
    skills_prompt_jd_answer = llm_2.invoke(skills_prompt_jd).content

    texts = [skills_prompt_resume_answer, skills_prompt_jd_answer]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(X)
    skills_score = round(similarities[0, 1]*100, 2)




    experience_prompt_resume = f"""
    Please act as a HR expert and tell me about experience of this candidate, by analyzing his/her resume.

    Here is the resume content of the candidate: ```{resume_content}```.

    Note: The output should be only experience
    """
    experience_prompt_resume_answer = llm_2.invoke(experience_prompt_resume).content

    experience_prompt_jd = f"""
    I am providing you a Job Description, Please act as a HR expert and tell me what and how much experience is required for this job.

    Here is the job description: ```{job_description}```.

    Note: The output should be only experience requirement
    """
    experience_prompt_jd_answer = llm_2.invoke(experience_prompt_jd).content

    texts = [experience_prompt_resume_answer, experience_prompt_jd_answer]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(X)
    experience_score = round(similarities[0, 1]*100, 2)



    education_prompt_resume = f"""
    Please act as a HR expert and tell me about education of this candidate, by analyzing his/her resume.

    Here is the resume content of the candidate: ```{resume_content}```.

    Note: The output should be only education (specifically about degree)
    """
    education_prompt_resume_answer = llm_2.invoke(education_prompt_resume).content

    education_prompt_jd = f"""
    I am providing you a Job Description, Please act as a HR expert and tell me what education is required for this job.

    Here is the job description: ```{job_description}```.

    Note: The output should be only education requirement (specifically about what degree is required)
    """
    education_prompt_jd_answer = llm_2.invoke(education_prompt_jd).content

    texts = [education_prompt_resume_answer, education_prompt_jd_answer]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(X)
    education_score = round(similarities[0, 1]*100, 2)

    categories = ['Experience', 'Skills', 'Education']
    scores = [experience_score, skills_score, education_score]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, scores, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.title('Scoring (Resume vs Job Description)', fontsize=16)
    plt.xlabel('Sections', fontsize=14)
    plt.ylabel('Scores (%)', fontsize=14)
    plt.ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}%', ha='center', va='bottom', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("static/chart.png")
    plt.show()

    return ai_response, experience_score, education_score, skills_score 



def func_select_best_candidates(resume_content, priority, job_description):
    score_template_string = """Please analyze this resume for the job description as: '{job_description}' and tell me if candidate is capable and ready for this role.
(Be a little strict in your decision by observing the job requirements specifically take the '{priority}' seriously) ,
please format you answer as 'Right Fit' or 'Not a Right Fit' in first line, then a score out of 100 (score/100) in the second line where not right fit do not exceed a score
more than 30 out of 100 and also if it is entirely different from resume_content make a score of 0 out of 100 (reflecting a match of candidate with job description) and then give explanation of the decision in the third line. Please be little rigid in formatting especially for score as example 80/100.
Note: If the score is above than 70 and below 90, then choose a random number between 70 to 90 because I want to sort candidates based on score
 RESUME CONTENT: {resume_content} {score_format_instructions}
"""
    score_prompt_template = ChatPromptTemplate.from_template(score_template_string)
    score = ResponseSchema(name="score", description="this will be the score allocated to candiadate based on resume (return Only obtained score in Number. e.g. 85 (Dont return like 85/100))")
    why_score = ResponseSchema(name="why_score", description="this will be the explanation and reason of why candiadate has got that score (Only explanation)")
    score_response_schema = [score, why_score]
    score_schema_output_score = StructuredOutputParser.from_response_schemas(score_response_schema)
    score_format_instructions = score_schema_output_score.get_format_instructions()
    score_messages = score_prompt_template.format_messages(resume_content=resume_content,  priority=priority, job_description=job_description, score_format_instructions=score_format_instructions)
    score_response = llm(score_messages)
    score_response_as_dict = score_schema_output_score.parse(score_response.content)
    score = score_response_as_dict["score"]
    why_score = score_response_as_dict["why_score"]

    return score, why_score


# making route for homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route('/job_role', methods=['GET', 'POST'])
def job_role():
    print("job role")
    return render_template("job_role.html")

@app.route('/job_desc', methods=['GET', 'POST'])
def job_desc():
    return render_template("job_desc.html")

@app.route('/select_best', methods=['GET', 'POST'])
def select_best():
    return render_template("select_best.html")


@app.route('/analyze_for_job_role', methods=['GET', 'POST'])
def analyze_for_job_role():
    job_role = request.form["job_role"]
    resume_file = request.files["resume_file"]
    # fhandle = open(resume_file_path, 'rb')
    pdfReader = PyPDF2.PdfReader(resume_file)
    number_of_pages = len(pdfReader.pages)
    text = ""
    for page_number in range(number_of_pages):   # use xrange in Py2
        page = pdfReader.pages[page_number]
        page_content = page.extract_text()
        text += page_content
    # Strip out unwanted text
    text = text.replace('o ','')
    resume_content = text.replace('|', '')
    ai_answer = func_analyze_for_job_role(resume_content, job_role)

    return render_template("display_result.html", ai_answer=ai_answer)


@app.route('/analyze_for_job_desc', methods=['GET', 'POST'])
def analyze_for_job_desc():
    job_description = request.form["job_description"]
    priority = request.form["priority"]
    resume_file = request.files["resume_file"]
    # fhandle = open(resume_file_path, 'rb')
    pdfReader = PyPDF2.PdfReader(resume_file)
    number_of_pages = len(pdfReader.pages)
    text = ""
    for page_number in range(number_of_pages):   # use xrange in Py2
        page = pdfReader.pages[page_number]
        page_content = page.extract_text()
        text += page_content
    # Strip out unwanted text
    text = text.replace('o ','')
    resume_content = text.replace('|', '')
    ai_answer, experience_score, education_score, skills_score = func_analyze_for_job_desc(resume_content, job_description, priority)

    return render_template("display_result.html", ai_answer=ai_answer, experience_score=experience_score, education_score=education_score, skills_score=skills_score)

# making route for analyze
@app.route('/select_best_candidates', methods=['GET', 'POST'])
def select_best_candidates():
    resume_files = request.files.getlist('resume_files[]')
    job_description = request.form["job_description"]
    priority = request.form["priority"]
    num_of_candidates = request.form["num_of_candidates"]
    
    all_results = []

    for resume in resume_files:
        pdfReader = PyPDF2.PdfReader(resume)
        number_of_pages = len(pdfReader.pages)
        text = ""
        for page_number in range(number_of_pages):   # use xrange in Py2
            page = pdfReader.pages[page_number]
            page_content = page.extract_text()
            text += page_content
        # Strip out unwanted text
        text = text.replace('o ','')
        resume_content = text.replace('|', '')

        score, why_score = func_select_best_candidates(resume_content, priority, job_description)
        analysis_result = (int(score), resume.filename, why_score)
        print(analysis_result)
        print(" ")
        all_results.append(analysis_result)        
    print("All analysis DONE")

    # Sort the candidate scores based on the first element of each tuple (i.e., the score)
    sorted_candidates = sorted(all_results, reverse=True, key=lambda x: x[0])

    # Select the top 2 candidates
    top_candidates = sorted_candidates[:int(num_of_candidates)]

    return render_template("display_result.html", top_candidates=top_candidates)


# making route for use app again
@app.route('/use_app', methods=['GET', 'POST'])
def use_app():
    return render_template("index.html")

# for dealing with cache storing issue of browser
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True) 