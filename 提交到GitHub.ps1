# 智能炒股分析APP - 代码提交脚本
# 不需要Git客户端，直接通过GitHub API提交
# =============================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GitHub 代码提交工具 (无需Git客户端)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 仓库配置
$REPO = "Idol993/rh001"
$ROUND = 1

# 读取Token
if ($env:GITHUB_TOKEN) {
    Write-Host "[提示] 检测到环境变量 GITHUB_TOKEN" -ForegroundColor Green
    $TOKEN = $env:GITHUB_TOKEN
} else {
    Write-Host "请输入 GitHub Token (需要 repo 权限):" -ForegroundColor Yellow
    Write-Host "  获取地址: https://github.com/settings/tokens" -ForegroundColor Gray
    $secureToken = Read-Host "  Token" -AsSecureString
    $TOKEN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    )
}

if ([string]::IsNullOrWhiteSpace($TOKEN)) {
    Write-Host "[错误] Token不能为空" -ForegroundColor Red
    exit 1
}

# 读取提交信息
Write-Host ""
Write-Host "请输入提交说明 (直接回车使用默认):" -ForegroundColor Yellow
$message = Read-Host "  说明"
if ([string]::IsNullOrWhiteSpace($message)) {
    $message = "round-$ROUND`: 提交代码 - 智能炒股分析APP核心逻辑 (多因子特征工程 + 时序预测模型 + 跨端代码生成)"
}

Write-Host ""
Write-Host "仓库: $REPO" -ForegroundColor Cyan
Write-Host "信息: $message" -ForegroundColor Cyan
Write-Host ""
Write-Host "开始提交..." -ForegroundColor Green

# 设置项目根目录
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

# 查找Python
$pythonCmd = $null
$pythonPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "$env:ProgramFiles\Python310\python.exe"
)

foreach ($p in $pythonPaths) {
    if (Test-Path $p) {
        $pythonCmd = $p
        break
    }
}

if (-not $pythonCmd) {
    try {
        $testPython = & python --version 2>&1
        $pythonCmd = "python"
    } catch {
        try {
            $testPython = & py --version 2>&1
            $pythonCmd = "py"
        } catch {
            Write-Host "[错误] 未找到Python，请先安装Python 3.8+" -ForegroundColor Red
            Write-Host "  下载地址: https://www.python.org/downloads/" -ForegroundColor Gray
            exit 1
        }
    }
}

Write-Host "Python命令: $pythonCmd" -ForegroundColor Gray

# 执行提交
$scriptPath = Join-Path $PSScriptRoot "python\push_to_github.py"
& $pythonCmd $scriptPath --token $TOKEN --repo $REPO --message $message --round $ROUND --path $PROJECT_ROOT

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  提交成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  仓库地址: https://github.com/$REPO" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  提交失败，错误代码: $exitCode" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
