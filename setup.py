from setuptools import setup

setup(
    name='broker-spec-gpt',
    version='',
    packages=[''],
    url='',
    license='',
    author='Poul Hornsleth',
    author_email='',
    description='',
    install_requires=[
        'openai',
        'python-dotenv',
        'langchain',
        'streamlit',
        'PyPDF2',
        'watchdog',
        'transformers',
        'torch',
        'InstructorEmbedding',
        'sentence_transformers',
        'faiss-cpu'

       # 'htmlTemplates',
    ]
)
