$ErrorActionPreference = 'Stop'
$WebRoot = Split-Path $PSScriptRoot -Parent
$ModelRoot = Join-Path $WebRoot 'public\models'
New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $ModelRoot 'wasm'))) {
    Copy-Item -LiteralPath (Join-Path $WebRoot 'node_modules\@mediapipe\tasks-vision\wasm') -Destination (Join-Path $ModelRoot 'wasm') -Recurse
}
$Downloads = @{
    'pose_landmarker_lite.task' = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    'face_landmarker.task' = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    'hand_landmarker.task' = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
}
$Hashes = @{
    'pose_landmarker_lite.task' = '59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A'
    'face_landmarker.task' = '64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF'
    'hand_landmarker.task' = 'FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1'
}
foreach ($Entry in $Downloads.GetEnumerator()) {
    $Target = Join-Path $ModelRoot $Entry.Key
    if (-not (Test-Path -LiteralPath $Target)) {
        Write-Host ('Downloading ' + $Entry.Key)
        $Partial = $Target + '.download'
        Invoke-WebRequest -Uri $Entry.Value -OutFile $Partial -UseBasicParsing
        if ((Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash -ne $Hashes[$Entry.Key]) {
            throw ('Model download checksum mismatch: ' + $Partial)
        }
        Move-Item -LiteralPath $Partial -Destination $Target
    }
    if ((Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash -ne $Hashes[$Entry.Key]) {
        throw ('Existing model checksum mismatch; inspect this file before retrying: ' + $Target)
    }
}
