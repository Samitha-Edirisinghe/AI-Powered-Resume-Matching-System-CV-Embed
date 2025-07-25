# Coding by Samitha Randika | https://www.linkedin.com/in/samitha-randika-edirisinghe-b3a68a2b6 #
# Configuration for datasets and models
DATASETS = [
    "bwbayu/job_cv_supervised",
    "cnamuangtoun/resume-job-description-fit",
    "InferencePrince555/Resume-Dataset",
    "jog-description-and-salary-in-indonesia",
    "itjobpostdescriptions",
    "resume-dataset"
]

MODEL_CONFIG = {
    "active_model": "sbert",
    "sbert_path": "trained_models/sbert",
    "glove_path": "trained_models/glove.model",
    "doc2vec_path": "trained_models/doc2vec.model",
    "job_taxonomy": [
        "System Administrator",
        "Database Administrator",
        "Web Developer",
        "Security Analyst",
        "Network Administrator",
        "Data Scientist",
        "DevOps Engineer",
        "Cloud Engineer",
        "Machine Learning Engineer",
        "Software Engineer"
    ]
}