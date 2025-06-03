import os
from pathlib import Path

README = Path("README.md")
BADGE_START = "<!-- BADGES_START -->"
BADGE_END = "<!-- BADGES_END -->"

# Load values from environment
coverage = os.getenv("COVERAGE", "0")
pylint = os.getenv("PYLINT_SCORE", "0")
ml_score = os.getenv("ML_SCORE", "0")
ml_max = os.getenv("ML_MAX", "10")

# Generate badge markdown
badges_md = f"""
![Coverage](https://img.shields.io/badge/coverage-{coverage}%25-brightgreen)
![Pylint](https://img.shields.io/badge/pylint-{pylint}-yellowgreen)
![ML Test Score](https://img.shields.io/badge/test--score-{ml_score}%2F{ml_max}-blue)
"""

def update_readme():
    if not README.exists():
        print("README.md not found")
        return

    content = README.read_text()

    if BADGE_START not in content or BADGE_END not in content:
        print("Badge markers not found, adding section.")
        content += f"\n\n{BADGE_START}\n{badges_md}\n{BADGE_END}\n"
    else:
        # Replace between markers
        pre = content.split(BADGE_START)[0]
        post = content.split(BADGE_END)[1]
        content = f"{pre}{BADGE_START}\n{badges_md}\n{BADGE_END}{post}"

    README.write_text(content.strip() + "\n")
    print("README updated with test/lint/coverage badges.")

if __name__ == "__main__":
    update_readme()
