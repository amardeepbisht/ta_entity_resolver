# setup.py (to live at the repo root, alongside README.md, requirements.txt)

from setuptools import setup, find_packages

setup(
    name="ta_entity_resolver",
    version="0.1.0",
    description="Modular ER engine for Databricks",
    author="Your Name",
    author_email="you@example.com",

    # Tell setuptools that our packages are under the ta_entity_resolver/ folder
    package_dir={"": "ta_entity_resolver"},

    # Find every package (every folder with __init__.py) under ta_entity_resolver/
    packages=find_packages(where="ta_entity_resolver"),

    # Include data files declared in package_data (and in MANIFEST.in if you have one)
    include_package_data=True,

    # Your runtime dependencies
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

    # Optional console script entrypoint
    entry_points={
        "console_scripts": [
            "ta-er=ta_entity_resolver.main:main",
        ],
    },

    # Declare non-Python files to ship in your package
    package_data={
        # always bundle your config file
        "ta_entity_resolver.resolver_config": ["config.yaml"],

        # if you want to ship sample data (e.g. CSVs for quick-start), uncomment:
         "ta_entity_resolver.sample_data": ["**/*.csv"],
    },

    # Don’t compress the egg (so data files remain as-is on install)
    zip_safe=False,
)
