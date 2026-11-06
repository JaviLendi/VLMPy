#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{6F1B7C52-3A0D-4E4B-9C55-5D2E8A1B7F10}
AppName=VLMPy
AppVersion={#AppVersion}
AppPublisher=Javier Lendínez
DefaultDirName={localappdata}\Programs\VLMPy
DefaultGroupName=VLMPy
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\docs\img\icon.ico
UninstallDisplayIcon={app}\VLMPy.exe
LicenseFile=..\LICENSE
OutputDir=..\dist\installer
OutputBaseFilename=VLMPy-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\VLMPy\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\VLMPy"; Filename: "{app}\VLMPy.exe"
Name: "{autodesktop}\VLMPy"; Filename: "{app}\VLMPy.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VLMPy.exe"; Description: "{cm:LaunchProgram,VLMPy}"; Flags: nowait postinstall skipifsilent

; Data written next to the exe (saved configs/results) must survive uninstall.
[UninstallDelete]
Type: filesandordirs; Name: "{app}\temp_stl"
