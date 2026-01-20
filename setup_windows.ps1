
Write-Host "🔍 Checking for Python..."
try {
    $version = python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python not found" }
    Write-Host "✅ Found $version"
} catch {
    Write-Error "❌ Python is not installed or not in your PATH."
    Write-Host "Please install Python from https://python.org or the Microsoft Store."
    Write-Host "⚠️  IMPORTANT: During installation, check 'Add Python to PATH'."
    exit 1
}

Write-Host "📦 Creating virtual environment 'venv'..."
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to create virtual environment."
    exit 1
}

Write-Host "⬇️  Installing requirements..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to install requirements."
    exit 1
}

Write-Host "✅ Setup complete!"
Write-Host "🚀 To activate the environment, run:"
Write-Host "    .\venv\Scripts\Activate.ps1"
