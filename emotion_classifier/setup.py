from setuptools import setup, find_packages

setup(
    name='emotion_classifier',
    version='0.1.0',
    packages=find_packages(where="emotion_classifier"),
    package_dir={"": "emotion_classifier"},

    install_requires=[
        "torch>=2.0",
        "torchaudio>=2.0",
        "PyYAML>=5.4",
        "tensorboard",
        "matplotlib",
        "scikit-learn",
        "soundfile",
        "transformers",
    ],
    python_requires=">=3.8",
    author="Uttam Paudel",
    description="Emotion classification pipeline with SSL models and attention pooling",
)
