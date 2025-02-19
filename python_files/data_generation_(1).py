# -*- coding: utf-8 -*-
"""data_generation (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14A-zACeMo2BsPs7Qr2-aq3kmlVxiqCXb
"""

!pip search together
!pip install together

from together import Together
import random
import pandas as pd
import os
import time
import time

# Configure Together API

from together import Together
import os
#set your Together.ai API key
os.environ["TOGETHER_API_KEY"]="7f1fe73226e2d457b33631f1afc8366270899f50105ce2998870ce3774fb4397"
client=Together()

names = [
    "Arun", "Akaay", "Akshay", "Akansha", "Anika", "Ayushi", "Ankita", "Aarushi", "Anuj", "Aditya", "Ansh", "Aditi", "Aarav", "Amrita", "Avantika", "Arjun",
    "Bhuvi", "Bhairavi", "Chhavi", "Chirayu", "Dipali", "Divyansh", "Ekta", "Eshan", "Eklavya", "Eshar", "Falguni", "Farah", "Faria", "Gargi", "Gauri", "Gauransh",
    "Gauranshi", "Gita", "Girish", "Gagan", "Govind", "Gulshan", "Harshad", "Hritik", "Harsita", "Indu", "Isha", "Ishit", "Ivaan", "Inaan", "Janish",
    "Kadambari", "Kartik", "Kush", "Kusha", "Khushi", "Kunal", "Kavya", "Kritika", "Kiara", "Lavish", "Mansi", "Manish", "Manikant", "Mukesh", "Meena", "Meera",
    "Mannat", "Monika", "Nishant", "Nidhi", "Nishit", "Nishita", "Navya", "Ojasv", "Omkar", "Priyanshu", "Pooja", "Priyank", "Preeti", "Prem",
    "Rishabh", "Risika", "Rishi", "Riya", "Riyan", "Rita", "Ritika", "Rohit", "Samaira", "Sonia", "Subham", "Surbhi", "Subhi", "Samantha", "Samar", "Tushar",
    "Tulika", "Tanu", "Tisha", "Udisha", "Udit", "Vansh", "Virat", "Vaibhav", "Vivan", "Vaishali", "Vaidhayi", "Ved", "Vibhanshi", "Yash"
]

last_names = [
    "Sharma", "Dixit", "Chaudhary", "Bhardwaj", "Singh", "Raj", "Katiyan", "Sherawat", "Baliyan", "Advani", "Chaturvedhi", "Soam", "Modi", "Latiyan",
    "Bhardwaj", "Shandaliya", "Sharma", "Malik", "Pundir", "Chaudhary", "Dubey", "Tiwari", "Mishra", "Panday", "Bhatt", "Kapoor", "Raina", "Joshi", "Deshpande",
    "Tripathi", "Bhardwaj", "Panday", "Sharma", "Baliyan", "Chaudhary", "Kohl", "Singh", "Raj", "Chabbra", "Dutt", "Bali", "Vaid", "Dahiya", "Phogat", "Prabhu",
    "Arjun", "Bhardwaj", "Dubey", "Shukla", "Cheema", "Punia", "Jakhar", "Sihag", "Brar", "Latiyan", "Sharma", "Trivedi", "Muchhall", "Aryan", "Dhawan", "Oberoi",
    "Panday", "Bhardwaj", "Bhullar", "Manak", "Upadhyay", "Dixit", "Chauhan", "Dhiman", "Rathore", "Tomar", "Solanki", "Saxena", "Mathur", "Verma", "Jain", "Singhal",
    "Pal", "Malik", "Sharma", "Tiwari", "Singh", "Sisodia", "Yadav", "Goyal", "Mittal", "Maheshwari", "Bachhav", "Bansal", "Shukla", "Bhardwaj", "Shandaliya", "Dixit",
    "Aggarwal", "Gupta", "Nigam", "Soam", "Karna", "Mathur", "Arya"
]

designations = {
    "Data Analyst": {
        "skills": ["Excel", "SQL", "Python/R", "visualization (Tableau, Power BI)"],
        "domains": ["Business", "Market", "Finance"]
    },
    "Software Developer": {
        "skills": ["Advanced coding", "debugging", "system design knowledge"],
        "domains": ["E-Commerce", "Banking", "Retail", "Gaming", "Telecommunications"]
    },
    "DevOps Engineer": {
        "skills": ["Cloud platforms (AWS, Azure)", "CI/CD", "scripting"],
        "domains": ["Software Development", "Cloud Computing", "E-Commerce"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Network security", "threat analysis", "incident response"],
        "domains": ["Information Technology", "Retail", "Finance", "Banking", "Telecommunications"]
    },
    "Full-Stack Developer": {
        "skills": ["Frontend (HTML, CSS, JavaScript)", "Backend (Node.Js, Django, etc)", "database management"],
        "domains": ["E-Commerce", "Education and E-Learning", "Enterprise software", "Social media and marketing"]
    },
    "AI Engineer": {
        "skills": ["AI algorithms", "neural networks", "NLP"],
        "domains": ["Virtual Assistants", "Retail", "Cybersecurity", "Automation"]
    },
    "UI Designer": {
        "skills": ["User Experience (UX Design)", "Prototyping", "Graphic Design", "Interaction Design", "Frontend Development (HTML, CSS, JavaScript)"],
        "domains": ["Gaming", "Media", "Retail"]
    }
}

Experience_level = ["Entry-Level", "Associate-level", "Mid-Level", "Senior-Level", "Expert-Level"]
work_environment = ["Remote", "Hybrid", "On-site"]

# Predefined reason sets
reasons_for_selection = [
    "Demonstrated outstanding problem solving skills. The candidate demonstrated strong problem solving abilities and flexibilities in previous roles.",
    "Strong alignment with job requirements. The candidate's skills, experience, and qualifications are an excellent match for the role.",
    "Excellent Technical skills and expertise. The candidate's technical skills with relevant experience make a good fit.",
    "Proven leadership potential. The candidate showed clear leadership or managerial skills during the interview process.",
    "Effective communication and collaboration skills. The candidate demonstrated clear and effective communication skills.",
    "Strong cultural fit. The candidate's values, work ethic, and personality align with the company's culture.",
    "Ability to innovate and drive changes. The candidate displayed creativity and innovative thinking in problem-solving.",
    "Positive attitude and enthusiasm. The candidate displayed a positive attitude and enthusiasm during the interview process."
]

reasons_for_rejection = [
    "Lack of required skills. The candidate's qualifications did not meet the essential skills needed for the role.",
    "Stronger candidate pool. Other candidates demonstrated stronger qualifications or a better fit for the position.",
    "Lack of experience. The candidate's experience is too junior for the position.",
    "Overqualified for the role. The candidate's qualifications or experience exceed the requirements for the position, leading to concerns about job satisfaction or retention.",
    "Lack of required certifications or credentials. The candidate does not possess the necessary certifications or credentials required for the role.",
    "Poor interview performance. The candidate did not effectively demonstrate their skills or fit for the role during the interview process.",
    "Struggled to demonstrate soft skills. The candidate lacked key interpersonal skills, such as communication, adaptability, or teamwork."
]

languages_known = ["English", "Spanish", "Hindi", "French", "Tamil", "Telugu"]

# Function to generate a text response based on the given prompt
def generate_text(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": f"{prompt}"}],
    )
    return response.choices[0].message.content

# Generating data for the table
data = []
for i in range(500):  # the range can be adjusted as per the need
    id_formatted = f"nehaj{i+1:04d}"  # ID in the format nehaj01, nehaj02, etc.
    name = random.choice(names)
    last_name = random.choice(last_names)
    full_name = f"{name} {last_name}"
    designation = random.choice(list(designations.keys()))
    expected_experience = random.choice(["0-2 years", "3-5 years", "6-9 years", "10+ years"])

    # Ensure the job description matches the designation and correctly references "domains"
    domains_needed = ", ".join(designations[designation]["domains"])
    job_description = f"Expected_experience: {expected_experience}, Domains: {domains_needed}"

    # Determine if the candidate should be selected or rejected
    select_status = random.choice(["Select", "Reject"])

    # Randomize experience level and work environment
    experience_level = random.choice(Experience_level)
    work_environment = random.choice(work_environment)

    # Randomly select languages
    selected_languages = random.sample(languages_known, k=random.randint(1, len(languages_known)))

    # Add all this data to the list as a dictionary or tuple
    data.append({
        "ID": id_formatted,
        "Full Name": full_name,
        "Designation": designation,
        "Job Description": job_description,
        "Selection Status": select_status,
        "Experience Level": experience_level,
        "Work Environment": work_environment,
        "Languages Known": selected_languages
    })

# Now 'data' contains the generated candidates' information

# Select reasons for decision
reasons = random.sample(
    [
        "Demonstrated outstanding problem solving skills. The candidate demonstrated strong problem solving abilities and flexibilities in previous roles.",
        "Strong alignment with job requirements. The candidate's skills, experiences, and qualifications are an excellent match for the role.",
        "Excellent technical skills and expertise. The candidate's technical skills with relevant experience makes a good fit.",
        "Proven leadership potential. The candidate showed clear leadership or managerial skills during the interview process.",
        "Effective communication and collaboration skills. The candidate demonstrated clear and effective communication skills.",
        "Strong cultural fit. The candidate's values, work ethic, and personality align with the company's culture.",
        "Ability to innovate and drive changes. The candidate displayed creativity and innovative thinking in problem-solving.",
        "Positive attitude and enthusiasm. The candidate displayed a positive attitude and enthusiasm during the interview process."
    ]
    if select_status == "Select"
    else [
        "Lack of required skills. The candidate's qualifications did not meet the essential skills needed for the role.",
        "Stronger candidate pool. Other candidates demonstrated stronger qualifications or a better fit for the position.",
        "Lack of experience. The candidate's experience is too junior for the position.",
        "Overqualified for the role. The candidate's qualifications or experience exceed the requirements for the position, leading to concerns about job satisfaction or retention.",
        "Lack of required certifications or credentials. The candidate does not possess the necessary certifications or credentials required for the role.",
        "Poor interview performance. The candidate did not effectively demonstrate their skills or fit for the role during the interview process.",
        "Struggled to demonstrate soft skills. The candidate lacked key interpersonal skills, such as communication, adaptability, or teamwork."
    ],
    2
)

# Define prompt templates for transcripts, profiles
prompt_templates = {
    "Select": {
        "transcript": (
            f"'{job_description}'. Highlight strengths such as demonstrated skills in {', '.join(designations[designation]['skills'])}, "
            f"excellent problem-solving abilities, enthusiasm, and potential for growth. Emphasize their impressive experience in {domains_needed}. "
            "Do not generate generic statements like 'Here is an interview transcript for ...'."
        ),
        "profile": (
            f"Generate a positive, detailed profile for candidate {full_name} who is interviewing for the role of {designation}. "
            f"Highlight their strengths, such as a solid foundation in key skills like {', '.join(designations[designation]['skills'])}, "
            f"relevant domain experience in {', '.join(domains_needed)}, and enthusiasm to contribute. Mention any noteworthy achievements, "
            f"such as successful projects or past roles in similar domains. Mention their potential to excel and grow within the organization. "
            "Ensure that the profile is focused on showcasing their qualifications in the context of the job description. "
            "Do not generate generic statements like 'Here is a profile for ...'."
        ),
    },
    "Reject": {
        "transcript": (
            f"Generate a detailed, constructive interview transcript for candidate {full_name} interviewing for the role of {designation}. "
            f"Job description: '{job_description}'. Mention areas of improvement, such as lacking key technical skills in {', '.join(designations[designation]['skills'])}, "
            f"insufficient experience in the required domains, and limited understanding of job requirements. Highlight reasons like 'Struggled to communicate ideas' or 'Needed improvement in problem-solving skills'. "
            "Do not generate generic statements like 'Here is an interview transcript for ...'."
        ),
        "profile": (
            f"Generate a detailed profile for candidate {full_name} who is interviewing for the role of {designation}. "
            f"Focus on areas where the candidate's skills did not align with the job requirements, such as lacking proficiency in {', '.join(designations[designation]['skills'])}, "
            f"or insufficient experience in {', '.join(domains_needed)}. Mention any concerns raised during the interview, such as difficulty with certain tasks or lack of familiarity with specific tools or technologies. "
            "Despite these gaps, acknowledge their potential for growth if given further training or exposure. "
            "Ensure that the profile emphasizes constructive feedback while being respectful. "
            "Do not generate generic statements like 'Here is a profile for ...'."
        ),
    }
}

# Construct prompts based on the selection status
prompt_transcript = prompt_templates[select_status]["transcript"].format(name=full_name, designation=designation, job_description=", ".join(job_description))
prompt_profile = prompt_templates[select_status]["profile"].format(name=full_name, designation=designation, job_description=", ".join(job_description), experience_level=experience_level, work_environment=work_environment, languages_known=", ".join(selected_languages), expected_experience=expected_experience, domains_needed=domains_needed)
# Generate the transcript and profile
transcript = generate_text(prompt_transcript)
time.sleep(5)
profile = generate_text(prompt_profile)
time.sleep(5)
# Add to dataset
data.append({
"Transcript": transcript,
"Profile": profile,
"Select/Reject": select_status,
"Job Description": job_description,
"Reason for Decision": ", ".join(reasons),
})

# Convert to DataFrame and save as excel
df = pd.DataFrame(data)
print(df)
df.to_excel("Interview_candidates4.xlsx", index=False)







