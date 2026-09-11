$ErrorActionPreference = 'Stop'
$WebRoot = Split-Path $PSScriptRoot -Parent
$SampleRoot = Join-Path $WebRoot 'public\motions'
New-Item -ItemType Directory -Path $SampleRoot -Force | Out-Null
# The official Three.js FBX example attributes this character/motion to Mixamo.
# Download only for this local research demo; these are not relicensed as MIT.
$Target = Join-Path $SampleRoot 'samba.fbx'
$ExpectedHash = 'B9003EE562C87BF03051C3A502411B0808D3513F1D74A2011F7530D9F067069F'
if (-not (Test-Path -LiteralPath $Target)) {
    Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/mrdoob/three.js/r185/examples/models/fbx/Samba%20Dancing.fbx' -OutFile ($Target + '.download') -UseBasicParsing
    if ((Get-FileHash -LiteralPath ($Target + '.download') -Algorithm SHA256).Hash -ne $ExpectedHash) {
        throw 'Downloaded sample checksum mismatch. The file was left as samba.fbx.download for inspection.'
    }
    Move-Item -LiteralPath ($Target + '.download') -Destination $Target
}
if ((Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash -ne $ExpectedHash) {
    throw 'Existing samba.fbx does not match the pinned Three.js r185 sample. It was not modified.'
}
Write-Host ('Mixamo sample ready: ' + $Target)
