from setuptools import setup, find_packages

setup(
    name="mamba-video",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    author_email="",
    description="Attention-to-SSM architecture surgery for CPU-native video generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IshCPU-VideoGenLab/mamba-video",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.19.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "quality": ["torchmetrics>=1.2.0"],
        "viz": ["matplotlib>=3.7.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "mamba-video=mamba_video.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
