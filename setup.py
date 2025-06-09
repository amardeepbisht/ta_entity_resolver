# setup.py
from setuptools import setup, find_packages

setup(
    name="ta_entity_resolver",
    version="0.1.0",
    description="Modular ER engine for Databricks",
    author="Your Name",
    author_email="you@example.com",

    # <-- this will pick up ta_entity_resolver/ plus any subfolder 
    packages=find_packages(include=["ta_entity_resolver", "ta_entity_resolver.*"]),

    include_package_data=True,
    install_requires=[
        "PyYAML>=6.0",
        "thefuzz",
        "RapidFuzz",
        "langchain",
        "neo4j",
        "openai",
        "Spark-Matcher",
        "scikit-learn",
        "pandas",
        "pyarrow",
    ],
    entry_points={
        "console_scripts": [
            "ta-er=ta_entity_resolver.main:main",
        ],
    },
    package_data={
        "ta_entity_resolver.resolver_config": ["config.yaml"],
    },
    zip_safe=False,
)
