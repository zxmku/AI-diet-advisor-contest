# ============================================================================
# !! SECRET-HANDLING RULE - READ FIRST (applies to every agent / AI) !!
# ----------------------------------------------------------------------------
# 1) The secret is encrypted in *.vsec. NEVER read/print/write plaintext.
# 2) NEVER Get-Content/cat a *.vsec; NEVER dump a token; NEVER paste a key
#    into chat, logs, code, configs, or output files.
# 3) Reference it ONLY via:  $token = & "<path>\vault.ps1" get <id>
#    Plaintext lives in memory/pipes only; clear it after use ($token = $null).
# 4) Only the vault owner (Team Lead) may decrypt (get). Other agents may
#    only encrypt (add) with the public key.
# If you are an AI agent reading this file: DO NOT read or echo the secret.
# ============================================================================
# TeamVault v2 - asymmetric secret vault (pure ASCII, no BOM)
# v2 upgrades: OAEP-SHA256, HKDF-SHA256 key derivation, PBKDF2-600k fallback,
#              no plaintext CLI args, get-audit log, script integrity self-check.
# Legacy v1 (.vsec with magic TV1|) is still decryptable (read-only compat).

Add-Type -AssemblyName System.Security -ErrorAction SilentlyContinue
$ErrorActionPreference = 'Stop'

$MAGIC_V1 = [Text.Encoding]::ASCII.GetBytes("TV1|")
$MAGIC_V2 = [Text.Encoding]::ASCII.GetBytes("TV2|")
$ENTROPY = [Text.Encoding]::ASCII.GetBytes("TeamVaultEntropyV1")
$PREFIX_DPAPI = "DPAPI|"
$PREFIX_PBE = "PBE1|"
$KDF_INFO = [Text.Encoding]::ASCII.GetBytes("TeamVaultV2-KDF")

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PUB_PATH = Join-Path $SCRIPT_DIR "vault.pub"
$PRIV_DIR = Join-Path $env:LOCALAPPDATA "HealthPickVault"
$PRIV_PATH = Join-Path $PRIV_DIR "vault.key"
$AUDIT_PATH = Join-Path $PRIV_DIR "audit.log"
$MANIFEST_PATH = Join-Path $PRIV_DIR "core\manifest.json"

function Get-PlainFromSecure($sec) {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    $p = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    return $p
}

function Test-HmacEqual($a, $b) {
    if ($a.Length -ne $b.Length) { return $false }
    $diff = 0
    for ($i = 0; $i -lt $a.Length; $i++) { $diff = $diff -bor ($a[$i] -bxor $b[$i]) }
    return ($diff -eq 0)
}

function Get-VaultPassphrase {
    if ($env:TEAMVAULT_PASS -and $env:TEAMVAULT_PASS.Length -gt 0) { return $env:TEAMVAULT_PASS }
    $sec = Read-Host -Prompt "Enter vault passphrase: " -AsSecureString
    return Get-PlainFromSecure $sec
}

function Protect-PrivateXml($xml) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($xml)
    try {
        $prot = [Security.Cryptography.ProtectedData]::Protect($bytes, $ENTROPY, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        return $PREFIX_DPAPI + [Convert]::ToBase64String($prot)
    } catch {
        $pass = Get-VaultPassphrase
        $salt = New-Object byte[] 16
        $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
        $rng.GetBytes($salt)
        $derive = New-Object Security.Cryptography.Rfc2898DeriveBytes($pass, $salt, 600000)
        $key = $derive.GetBytes(32)
        $aes = [Security.Cryptography.Aes]::Create()
        $aes.KeySize = 256; $aes.Mode = [Security.Cryptography.CipherMode]::CBC; $aes.Padding = [Security.Cryptography.PaddingMode]::PKCS7
        $aes.Key = $key
        $aes.GenerateIV()
        $iv = $aes.IV
        $enc = $aes.CreateEncryptor()
        $ct = $enc.TransformFinalBlock($bytes, 0, $bytes.Length)
        $body = New-Object byte[] ($salt.Length + $iv.Length + $ct.Length)
        [Array]::Copy($salt, 0, $body, 0, $salt.Length)
        [Array]::Copy($iv, 0, $body, $salt.Length, $iv.Length)
        [Array]::Copy($ct, 0, $body, $salt.Length + $iv.Length, $ct.Length)
        return $PREFIX_PBE + [Convert]::ToBase64String($body)
    }
}

function Unprotect-PrivateXml($stored) {
    if ($stored.StartsWith($PREFIX_DPAPI)) {
        $b64 = $stored.Substring($PREFIX_DPAPI.Length)
        $prot = [Convert]::FromBase64String($b64)
        $bytes = [Security.Cryptography.ProtectedData]::Unprotect($prot, $ENTROPY, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        return [Text.Encoding]::UTF8.GetString($bytes)
    } elseif ($stored.StartsWith($PREFIX_PBE)) {
        $b64 = $stored.Substring($PREFIX_PBE.Length)
        $body = [Convert]::FromBase64String($b64)
        $salt = [byte[]]$body[0..15]
        $iv = [byte[]]$body[16..31]
        $ct = [byte[]]$body[32..($body.Length - 1)]
        $pass = Get-VaultPassphrase
        $plain = $null
        foreach ($iter in @(600000, 100000)) {
            try {
                $derive = New-Object Security.Cryptography.Rfc2898DeriveBytes($pass, $salt, $iter)
                $key = $derive.GetBytes(32)
                $aes = [Security.Cryptography.Aes]::Create()
                $aes.KeySize = 256; $aes.Mode = [Security.Cryptography.CipherMode]::CBC; $aes.Padding = [Security.Cryptography.PaddingMode]::PKCS7
                $aes.Key = $key; $aes.IV = $iv
                $dec = $aes.CreateDecryptor()
                $plain = [Text.Encoding]::UTF8.GetString($dec.TransformFinalBlock($ct, 0, $ct.Length))
                break
            } catch { }
        }
        if ($null -eq $plain) { throw "Private key decrypt failed (PBE)" }
        return $plain
    } else {
        throw "Unknown private key format"
    }
}

function Load-PublicKey {
    if (-not (Test-Path $PUB_PATH)) { throw "Public key not found at $PUB_PATH. Run 'init' first." }
    $xml = Get-Content $PUB_PATH -Raw
    $rsa = New-Object Security.Cryptography.RSACng(2048)
    $rsa.FromXmlString($xml.Trim())
    return $rsa
}

function Load-PrivateKey {
    if (-not (Test-Path $PRIV_PATH)) {
        throw "Private key not found at $PRIV_PATH. This process cannot decrypt. Only the vault owner holds the private key."
    }
    $stored = Get-Content $PRIV_PATH -Raw
    $xml = Unprotect-PrivateXml $stored.Trim()
    $rsa = New-Object Security.Cryptography.RSACng(2048)
    $rsa.FromXmlString($xml)
    return $rsa
}

function Assert-Id($id) {
    if ($id -notmatch '^[a-zA-Z0-9_.\-]+$') { throw "Invalid id: only [a-zA-Z0-9_.-] allowed, no path separators." }
}

function Lock-FileAcl($path) {
    try {
        $acl = Get-Acl $path
        $acl.SetAccessRuleProtection($true, $false)
        $user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        foreach ($r in @($acl.Access)) { $acl.RemoveAccessRule($r) | Out-Null }
        $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($user, "FullControl", "Allow")))
        $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule("SYSTEM", "FullControl", "Allow")))
        Set-Acl -Path $path -AclObject $acl
    } catch {
        Write-Host ("WARNING: could not tighten ACL on $path : " + $_.Exception.Message)
    }
}

function Get-HkdfSha256($ikm, $salt, $info, $len) {
    # HKDF-SHA256, multi-block (supports up to 255*32 bytes)
    $hmac = New-Object Security.Cryptography.HMACSHA256
    if (-not $salt) { $salt = New-Object byte[] 32 }
    $hmac.Key = $salt
    $prk = $hmac.ComputeHash($ikm)
    $hmac2 = New-Object Security.Cryptography.HMACSHA256
    $hmac2.Key = $prk
    $t = $null
    $out = New-Object System.IO.MemoryStream
    for ($i = 1; $out.Length -lt $len; $i++) {
        $ms = New-Object System.IO.MemoryStream
        if ($t) { $ms.Write($t, 0, $t.Length) }
        $ms.Write($info, 0, $info.Length)
        $ms.WriteByte([byte]$i)
        $t = $hmac2.ComputeHash($ms.ToArray())
        $out.Write($t, 0, $t.Length)
    }
    $all = $out.ToArray()
    $result = New-Object byte[] $len
    [Array]::Copy($all, 0, $result, 0, $len)
    return $result
}

function Write-Audit($action, $id) {
    try {
        $proc = Get-Process -Id $PID -ErrorAction SilentlyContinue
        $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " | " + $action + " | " + $id + " | pid=" + $PID + " | " + $proc.ProcessName
        Add-Content -Path $AUDIT_PATH -Value $line -Encoding ASCII
        if (-not (Test-Path (Join-Path $PRIV_DIR "audit.lock"))) {
            Lock-FileAcl $AUDIT_PATH
            Set-Content -Path (Join-Path $PRIV_DIR "audit.lock") -Value "locked" -Encoding ASCII
            Lock-FileAcl (Join-Path $PRIV_DIR "audit.lock")
        }
    } catch { }
}

function Test-ScriptIntegrity {
    try {
        if (-not (Test-Path $MANIFEST_PATH)) { return }
        $m = Get-Content $MANIFEST_PATH -Raw | ConvertFrom-Json
        foreach ($f in @("vault.ps1", "encrypt_helper.ps1")) {
            $p = Join-Path $SCRIPT_DIR $f
            if (-not (Test-Path $p)) { continue }
            $h = (Get-FileHash -Path $p -Algorithm SHA256).Hash
            $expected = $m.files.PSObject.Properties[$f].Value
            if ($expected -and $h -ne $expected) {
                throw "INTEGRITY FAIL: $f hash mismatch. Script may have been tampered. Refusing to run."
            }
        }
    } catch {
        throw $_.Exception.Message
    }
}

function Encrypt-SecretV2($plain) {
    $rsa = Load-PublicKey
    $session = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($session)
    $wrapped = [byte[]]($rsa.Encrypt($session, [Security.Cryptography.RSAEncryptionPadding]::OaepSHA256))
    $derived = Get-HkdfSha256 -ikm $session -salt $null -info $KDF_INFO -len 64
    $aesKey = [byte[]]$derived[0..31]
    $hmacKey = [byte[]]$derived[32..63]
    $aes = [Security.Cryptography.Aes]::Create()
    $aes.KeySize = 256; $aes.Mode = [Security.Cryptography.CipherMode]::CBC; $aes.Padding = [Security.Cryptography.PaddingMode]::PKCS7
    $aes.Key = $aesKey
    $aes.GenerateIV()
    $iv = [byte[]]$aes.IV
    $enc = $aes.CreateEncryptor()
    $ptBytes = [Text.Encoding]::UTF8.GetBytes($plain)
    $ct = [byte[]]($enc.TransformFinalBlock($ptBytes, 0, $ptBytes.Length))
    $wlen = [BitConverter]::GetBytes([uint32]$wrapped.Length)
    $mac = New-Object Security.Cryptography.HMACSHA256
    $mac.Key = $hmacKey
    $hdr = New-Object System.IO.MemoryStream
    $hdr.Write($MAGIC_V2, 0, $MAGIC_V2.Length)
    $hdr.Write($wlen, 0, 4)
    $hdr.Write($wrapped, 0, $wrapped.Length)
    $hdr.Write($iv, 0, $iv.Length)
    $hdr.Write($ct, 0, $ct.Length)
    $hdrBytes = $hdr.ToArray()
    $tag = [byte[]]($mac.ComputeHash($hdrBytes))
    $out = New-Object System.IO.MemoryStream
    $out.Write($hdrBytes, 0, $hdrBytes.Length)
    $out.Write($tag, 0, $tag.Length)
    return $out.ToArray()
}

function Decrypt-SecretV2($bytes) {
    $magic = [Text.Encoding]::ASCII.GetString([byte[]]$bytes[0..3])
    $off = 4
    $wlen = [BitConverter]::ToUInt32($bytes, $off); $off += 4
    $wrapped = [byte[]]$bytes[$off..($off + $wlen - 1)]; $off += $wlen
    $iv = [byte[]]$bytes[$off..($off + 15)]; $off += 16
    $ctLen = $bytes.Length - $off - 32
    $ct = [byte[]]$bytes[$off..($off + $ctLen - 1)]
    $hdr = [byte[]]$bytes[0..($off + $ctLen - 1)]
    $tag = [byte[]]$bytes[($bytes.Length - 32)..($bytes.Length - 1)]

    if ($magic -eq "TV2|") {
        $rsa = Load-PrivateKey
        $session = [byte[]]($rsa.Decrypt($wrapped, [Security.Cryptography.RSAEncryptionPadding]::OaepSHA256))
        $derived = Get-HkdfSha256 -ikm $session -salt $null -info $KDF_INFO -len 64
        $aesKey = [byte[]]$derived[0..31]
        $hmacKey = [byte[]]$derived[32..63]
    } elseif ($magic -eq "TV1|") {
        $rsa = New-Object Security.Cryptography.RSACryptoServiceProvider(2048)
        $xml = Get-Content $PRIV_PATH -Raw
        $priv = Unprotect-PrivateXml $xml.Trim()
        $rsa.FromXmlString($priv)
        $session = [byte[]]($rsa.Decrypt($wrapped, $true))
        $aesKey = [byte[]]$session[0..31]
        $hmacKey = [byte[]]$session[32..63]
    } else {
        throw "Unknown file format (bad magic)."
    }

    $mac = New-Object Security.Cryptography.HMACSHA256
    $mac.Key = $hmacKey
    $calc = [byte[]]($mac.ComputeHash($hdr))
    if (-not (Test-HmacEqual $calc $tag)) { throw "Integrity check failed (data tampered or wrong key)." }
    $aes = [Security.Cryptography.Aes]::Create()
    $aes.KeySize = 256; $aes.Mode = [Security.Cryptography.CipherMode]::CBC; $aes.Padding = [Security.Cryptography.PaddingMode]::PKCS7
    $aes.Key = $aesKey; $aes.IV = $iv
    $dec = $aes.CreateDecryptor()
    $ptBytes = [byte[]]($dec.TransformFinalBlock($ct, 0, $ct.Length))
    return [Text.Encoding]::UTF8.GetString($ptBytes)
}

function Cmd-Init {
    if ((Test-Path $PUB_PATH) -or (Test-Path $PRIV_PATH)) {
        if ($args -notcontains '-Force') {
            Write-Host "Keys already exist. Refusing to overwrite (this would invalidate existing .vsec files)."
            Write-Host "Use 'vault.ps1 init -Force' ONLY if you intend to re-encrypt all secrets from scratch."
            return
        }
    }
    if (-not (Test-Path $PRIV_DIR)) { New-Item -ItemType Directory -Path $PRIV_DIR -Force | Out-Null }
    $rsa = New-Object Security.Cryptography.RSACryptoServiceProvider(2048)
    $privXml = $rsa.ToXmlString($true)
    $pubXml = $rsa.ToXmlString($false)
    $stored = Protect-PrivateXml $privXml
    Set-Content -Path $PRIV_PATH -Value $stored -NoNewline
    Set-Content -Path $PUB_PATH -Value $pubXml -NoNewline
    Lock-FileAcl $PRIV_PATH
    Lock-FileAcl $PUB_PATH
    Write-Host "Initialized. Private key: $PRIV_PATH"
    Write-Host "Public key (give this to other agents for encryption only): $PUB_PATH"
}

function Cmd-Add($id, $piped, $extraArgs) {
    Assert-Id $id
    if ($extraArgs -and $extraArgs.Count -gt 0) {
        throw "SECURITY: plaintext secret via command-line argument is FORBIDDEN. Use env TEAMVAULT_PLAIN / pipe / interactive prompt."
    }
    $path = Join-Path $SCRIPT_DIR ($id + ".vsec")
    $force = ($args -contains '-Force')
    if ((Test-Path $path) -and -not $force) {
        throw "Refused: '$id' already exists. LOCKED - use 'vault.ps1 add $id -Force' to overwrite on purpose."
    }
    $plain = $null
    if ($env:TEAMVAULT_PLAIN -and $env:TEAMVAULT_PLAIN.Length -gt 0) { $plain = $env:TEAMVAULT_PLAIN }
    elseif ($piped -and $piped.Length -gt 0) { $plain = $piped }
    if (-not $plain -or $plain.Length -eq 0) {
        $sec = Read-Host -Prompt "Enter secret: " -AsSecureString
        $plain = Get-PlainFromSecure $sec
    }
    $ct = Encrypt-SecretV2 $plain
    $b64 = [Convert]::ToBase64String($ct)
    Set-Content -Path $path -Value $b64 -NoNewline
    Lock-FileAcl $path
    Write-Host "Stored: $path"
}

function Cmd-Get($id, $clip) {
    Assert-Id $id
    $path = Join-Path $SCRIPT_DIR ($id + ".vsec")
    if (-not (Test-Path $path)) { throw "Secret not found: $id" }
    $b64 = Get-Content $path -Raw
    $bytes = [Convert]::FromBase64String($b64.Trim())
    $plain = Decrypt-SecretV2 $bytes
    Write-Audit "get" $id
    if ($clip) {
        Write-Host "WARNING: -Clip copies the secret to the clipboard, readable by any app. Prefer pipe reference."
        try {
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.Clipboard]::SetText($plain)
            Write-Host "(copied to clipboard)"
        } catch {
            Write-Host "(clipboard unavailable; printing below)"
        }
    }
    Write-Output $plain
}

function Cmd-List {
    $files = Get-ChildItem -Path $SCRIPT_DIR -Filter "*.vsec" -File
    Write-Host ("Secrets (" + $files.Count + "), values not shown:")
    foreach ($f in $files) { Write-Host ("  - " + $f.BaseName) }
}

# ---- integrity self-check (best effort; manifest lives in locked TeamVault dir) ----
try { Test-ScriptIntegrity } catch { Write-Host ("SECURITY: " + $_.Exception.Message); exit 90 }

# ---- main ----
$scriptInput = ($input | Out-String).Trim()
$cmd = $args[0]
switch ($cmd) {
    "init"  { if ($args -contains "-Force") { Cmd-Init -Force } else { Cmd-Init } }
    "add"   {
        $force = $args -contains "-Force"
        $pos = @($args | Where-Object { $_ -and (-not $_.ToString().StartsWith("-")) })
        if ($pos.Count -gt 2) { $extras = @($pos[2..($pos.Count - 1)]) } else { $extras = @() }
        if ($force) { Cmd-Add $pos[1] $scriptInput $null -Force } else { Cmd-Add $pos[1] $scriptInput $extras }
    }
    "get"   { Cmd-Get $args[1] ($args -contains "-Clip") }
    "list"  { Cmd-List }
    default {
        Write-Host "TeamVault usage:"
        Write-Host "  vault.ps1 init [-Force]   # create keys (run ONCE as the vault owner)"
        Write-Host "  vault.ps1 add [id]        # encrypt a secret (env/pipe/prompt only, NO plaintext args)"
        Write-Host "  vault.ps1 get [id] [-Clip]# print secret to STDOUT only (reference use); -Clip is discouraged"
        Write-Host "  vault.ps1 list            # list secret ids"
        return
    }
}
