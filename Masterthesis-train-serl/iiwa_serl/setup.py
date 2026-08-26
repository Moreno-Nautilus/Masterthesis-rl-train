from setuptools import find_packages, setup


setup(
    name="iiwa_serl",
    version="0.1.0",
    description="Separate SERL route for KUKA iiwa insertion in Masterthesis-rl-train",
    packages=find_packages(),
    install_requires=[
        "absl-py",
        "flask",
        "gym>=0.26",
        "numpy",
        "opencv-python",
        "requests",
        "scipy",
    ],
)
