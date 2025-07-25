import os
import random
from utils.text_processing import clean_text

def create_doc2vec_documents(output_path, num_docs=500):
    """Create sample documents for Doc2Vec training"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample job descriptions
    templates = [
        "Experienced {role} with {experience} years in {skills} seeking position in {industry}",
        "Skilled {role} specializing in {skills} looking for opportunities in {domain}",
        "{role} with expertise in {skills} and experience with {tools}",
        "Professional {role} with background in {domain} and proficiency in {skills}",
        "Certified {role} with {experience}+ years experience in {industry} using {tools}"
    ]
    
    # Sample data for placeholders
    roles = ["software engineer", "data scientist", "devops engineer", 
             "web developer", "system administrator", "network engineer"]
    skills = ["Python", "Java", "JavaScript", "React", "AWS", "Docker", 
              "Kubernetes", "SQL", "machine learning", "data analysis"]
    tools = ["Git", "Jenkins", "Docker", "Kubernetes", "AWS", "Azure", 
             "TensorFlow", "PyTorch", "React", "Angular"]
    industries = ["tech", "finance", "healthcare", "e-commerce", "education"]
    domains = ["cloud computing", "AI development", "web applications", 
               "data engineering", "cybersecurity"]
    experiences = ["3", "5", "7", "10", "15"]
    
    # Generate documents
    documents = []
    for _ in range(num_docs):
        template = random.choice(templates)
        doc = template.format(
            role=random.choice(roles),
            skills=", ".join(random.sample(skills, random.randint(2, 4))),
            tools=", ".join(random.sample(tools, random.randint(1, 3))),
            industry=random.choice(industries),
            domain=random.choice(domains),
            experience=random.choice(experiences)
        )
        documents.append(clean_text(doc))
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + "\n")
    
    print(f"Created Doc2Vec documents at {output_path} with {num_docs} documents")
    return documents

if __name__ == "__main__":
    create_doc2vec_documents("../data/documents/documents.txt")