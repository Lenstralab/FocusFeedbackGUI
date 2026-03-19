!include "MUI2.nsh"

Name "FocusFeedbackGUI"
OutFile "FocusFeedbackGUI_setup.exe"

# define installation directory
InstallDir $LOCALAPPDATA\Programs\FocusFeedbackGUI

# Get installation folder from registry if available
InstallDirRegKey HKCU "Software\FocusFeedbackGUI" ""

# For removing Start Menu shortcut in Windows 7
RequestExecutionLevel user

!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

# start default section
Section

    # set the installation directory as the destination for the following actions
    SetOutPath $INSTDIR

    # Store installation folder
    WriteRegStr HKCU "Software\FocusFeedbackGUI" "" $INSTDIR

    # create the uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"

    # point the new shortcut at the program uninstaller
    createDirectory "$SMPROGRAMS\FocusFeedbackGUI"
    CreateShortcut "$SMPROGRAMS\FocusFeedbackGUI\FocusFeedbackGUI.lnk" "$INSTDIR\focusfeedbackgui\focusfeedbackgui.exe"
    CreateShortcut "$SMPROGRAMS\FocusFeedbackGUI\configuration.lnk" "$INSTDIR\focusfeedbackgui\_internal\focusfeedbackgui\conf.yml"
    CreateShortcut "$SMPROGRAMS\FocusFeedbackGUI\FocusFeedbackGUI Uninstall.lnk" "$INSTDIR\uninstall.exe"

    File /r "dist\focusfeedbackgui"

SectionEnd

# uninstaller section start
Section "uninstall"

    Delete "$INSTDIR\uninstall.exe"
    RMDir /r "$SMPROGRAMS\FocusFeedbackGUI"
    RMDir /r "$INSTDIR"

    DeleteRegKey /ifempty HKCU "Software\FocusFeedbackGUI"

# uninstaller section end
SectionEnd