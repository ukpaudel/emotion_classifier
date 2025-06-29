from setuptools import setup, find_packages

setup(
    name='emotion_classifier',
    version='0.1.0',
    packages=find_packages(),  # Automatically finds emotion_classifier/
    install_requires=[
        'torch',
        'torchaudio',
        'PyYAML'
    ],
)
