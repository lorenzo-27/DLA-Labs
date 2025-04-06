from setuptools import setup, find_packages

def parse_requirements(file_path):
    """Parse requirements from a file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

setup(
    name="DLA_Labs",
    version="1.0.0",
    author="Lorenzo Benedetti",
    description="DLA labs repository",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt")
)
