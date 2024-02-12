# resumeNebulaNavigator

The Resume Nebula Navigator is a powerful Streamlit application designed to streamline the process of analyzing resumes and tailoring them to specific job descriptions. Leveraging advanced natural language processing (NLP) techniques and machine learning models, this application provides comprehensive analysis, generates visualizations, and offers personalized recommendations to optimize resumes for targeted job opportunities.
Features

    File Upload: Users can seamlessly upload CVs and job descriptions in various formats, including PDF, DOCX, TXT, and image files (PNG, JPG, JPEG).
    Information Extraction: The application extracts text content from uploaded files, supporting structured and unstructured formats. It employs Optical Character Recognition (OCR) for image-based resumes to ensure comprehensive extraction.
    CV Analysis: Utilizes advanced language models to summarize CV information and extract keywords. Generates visually appealing word cloud visualizations to enhance keyword insights.
    Job Description Analysis: Extracts keywords from uploaded job descriptions to identify key requirements and preferences, facilitating alignment with resume content.
    Recommendation Generation: Delivers tailored recommendations based on thorough analysis of CVs and job descriptions, offering actionable insights to enhance resume quality and match job criteria effectively.

Installation and Setup

    Clone the repository:
git clone https://github.com/katoxiv/resumeNebulaNavigator.git
cd resume-nebula-navigator

Install dependencies:
create a virtual environment to avoid dependency issues.
pip install -r requirements.txt

Configure environment variables:
    Ensure all necessary environment variables are correctly configured, especially the OpenAI API key required for NLP tasks.

Run the application:
    streamlit run your_script.py

Usage:
    Access the application through the provided Streamlit URL.
    Upload your CV and job description files.
    Explore the extracted information, summarized CV details, keyword visualizations, and tailored recommendations.
    Optimize your resume based on the insights provided to enhance its compatibility with targeted job opportunities.

Contributing:
Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or submit a pull request.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Special thanks to OpenAI for providing the language model used in this application, and to the creators of the libraries and tools used to build and deploy it.

For more information, collaboration and insight, contact katoxiv@gmail.com.
