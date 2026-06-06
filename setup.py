import os

from setuptools import setup, find_packages


def _read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="whisper",
    py_modules=["whisper"],
    version="1.0",
    description="Whisper ASR with OpenVINO backend (configurable device: CPU/GPU/NPU)",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
