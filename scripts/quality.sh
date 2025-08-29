#!/bin/bash

# Quality check script for the RAG system
# Runs code formatting, linting, and style checks

set -e  # Exit on any error

echo "🔍 Running code quality checks..."

# Change to project root directory
cd "$(dirname "$0")/.."

# Run black formatting check
echo "📝 Checking code formatting with black..."
if ! uv run black backend/ --check --quiet; then
    echo "❌ Code formatting issues found. Run 'uv run black backend/' to fix."
    exit 1
fi
echo "✅ Code formatting looks good"

# Run isort import sorting check
echo "📦 Checking import sorting with isort..."
if ! uv run isort backend/ --check --quiet; then
    echo "❌ Import sorting issues found. Run 'uv run isort backend/' to fix."
    exit 1
fi
echo "✅ Import sorting looks good"

# Run flake8 linting
echo "🔍 Running linting with flake8..."
if ! uv run flake8 backend/; then
    echo "❌ Linting issues found. Please fix the issues above."
    exit 1
fi
echo "✅ Linting passed"

echo "🎉 All quality checks passed!"