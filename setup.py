from setuptools import setup, find_packages

setup(
    name="ta_entity_resolver",
    version="0.1.0",
    description="Modular ER engine for Databricks",
    packages=find_packages(),
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
            "ta-er=main:main",
        ],
    },
    package_data={
        "resolver_config": ["config.yaml"],
    },
)
