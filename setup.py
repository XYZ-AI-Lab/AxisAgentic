# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="axis-agentic",
    version="0.1.0",
    packages=find_packages(include=["agentic", "agentic.*", "recipe", "recipe.*"]),
    include_package_data=True,
    package_data={
        "agentic": ["config/*.yaml", "model_assets/*.yaml"],
        "recipe.web_search": ["configs/*.yaml"],
        "recipe.wide_search": ["configs/*.yaml"],
    },
    install_requires=[
        "aiofiles",
        "httpx",
        "json-repair",
        "jsonschema",
        "openai>=1.0",
        "pydantic>=2.0",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pre-commit", "pytest", "ruff"],
        "dashboard": ["altair", "pandas", "plotly", "streamlit"],
        "inference": ["datasets", "huggingface-hub", "pandas", "tqdm", "transformers"],
        "sandbox": ["e2b-code-interpreter"],
        "wide_search": ["dateparser", "numpy", "pandas"],
    },
    python_requires=">=3.12",
    description="Extensible execution and trajectory-collection framework for long-horizon AI agents.",
    license="Apache-2.0",
    author="XYZ Agentic Team",
    url="https://github.com/XYZ-AI-Lab/AxisAgentic",
    project_urls={
        "Documentation": "https://github.com/XYZ-AI-Lab/AxisAgentic#documentation",
        "Technical Report": "https://xyz-lab.ai/blogs/ai4ai-at-scale/",
    },
)
