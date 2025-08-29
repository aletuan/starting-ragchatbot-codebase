#!/bin/bash

# Code formatting script for the RAG system
# Automatically formats code using black and isort

set -e  # Exit on any error

echo "🎨 Formatting code..."

# Change to project root directory
cd "$(dirname "$0")/.."

# Run black formatting
echo "📝 Formatting code with black..."
uv run black backend/
echo "✅ Black formatting complete"

# Run isort import sorting
echo "📦 Sorting imports with isort..."
uv run isort backend/
echo "✅ Import sorting complete"

echo "🎉 Code formatting complete!"