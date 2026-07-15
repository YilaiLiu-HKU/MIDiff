from pathlib import Path

from setuptools import setup


def read_requirements():
    req_path = Path(__file__).with_name("requirements.txt")
    return [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="midiff",
    version="0.1.0",
    packages=[],
    description="Dependency metadata for the script-based MIDiff release.",
    install_requires=read_requirements(),
)
