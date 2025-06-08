# setup.py
from setuptools import setup, find_packages

def load_requirements(path="requirements.txt"):
    with open(path, encoding="utf-8") as f:
        # strip out any comments or blank lines
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="ta_entity_resolver",
    version="0.1.0",
    description="Modular ER engine for Databricks",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements(),    # ← loads every line in requirements.txt
    entry_points={
        "console_scripts": [
            "ta-er=main:main",
        ],
    },
    package_data={
        "resolver_config": ["config.yaml"],
    },
)
