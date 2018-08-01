#!/usr/bin/env bash

# Requires pandoc v2.2.1 or greater.

# Use the pandoc utility to convert the document from .odt format to markdown (.md) format appropriate for Github
# pandoc -s --from=odt --to=markdown_github --output=ReadMe.md ReadMe.odt

# Use the pandoc utility to convert the document from LaTex (.tex) format to markdown (.md) format appropriate for Github
pandoc \
--standalone \
--webtex=https://latex.codecogs.com/svg.latex? \
--atx-headers \
--mathjax \
--toc \
--top-level-division=chapter \
--number-offset=2 \
ReadMe.yaml \
--standalone \
--indented-code-classes=cpp \
--highlight-style=breezedark \
--listings \
--from=latex+yaml_metadata_block+tex_math_dollars \
--to=gfm+smart \
--output=ReadMe.md \
ReadMe.tex
