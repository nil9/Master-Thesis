#Region ;**** Directives created by AutoIt3Wrapper_GUI ****
#AutoIt3Wrapper_Icon=C:Program Files (x86)AutoIt3Iconsau3.ico
#EndRegion ;**** Directives created by AutoIt3Wrapper_GUI ****
; This code is taken from online source
#include <array.au3>
#include <ButtonConstants.au3>
#include <Date.au3>
#include <GUIConstantsEx.au3>
#include <MsgBoxConstants.au3>
#include <StaticConstants.au3>
#include <StringConstants.au3>
#include <Timers.au3>
#include <WindowsConstants.au3>
$gbTesting = True
; Create a GUI with various controls.
Global $hGUI = GUICreate("Example", 300, 200)
Global $iEvenOdd = 1
Global $gSeconds = 10
Global $giCalc = 1000 * $gSeconds
Global $aPos, $aPos2
Global $iTotal = 0
Global $iSkipped = 0
Global $giMouseAdjustment = 1

#Region ### START Koda GUI section ### Form=
$hGUI = GUICreate("Wiggle", 206, 148, 192, 125)
$idCheckbox = GUICtrlCreateCheckbox("On/Off", 41, 8, 50, 17)
$idCloser = GUICtrlCreateButton("Close", 15, 90, 84, 25)
$Label1 = GUICtrlCreateLabel("x:", 36, 32, 45, 17, $SS_RIGHT)
$Label2 = GUICtrlCreateLabel("y:", 36, 56, 45, 17, $SS_RIGHT)
$Label3 = GUICtrlCreateLabel("Secs", 148, 8, 39, 17, $SS_RIGHT)
$Label4 = GUICtrlCreateLabel("Total", 116, 32, 70, 17, $SS_RIGHT)
$Label5 = GUICtrlCreateLabel("Skipped", 116, 56, 70, 17, $SS_RIGHT)
$Label6 = GUICtrlCreateLabel("+/-", 116, 80, 70, 17, $SS_RIGHT)
$Label7 = GUICtrlCreateLabel("Label7", 13, 120, 180, 17)
GUICtrlSetData(-1,_now())
GUISetState(@SW_SHOW)
#EndRegion ### END Koda GUI section ###

; Display the GUI.
GUISetState(@SW_SHOW, $hGUI)

Example()

Func Example()
; Loop until the user exits.
Local $array, $iSecValue, $bLoop = True

While $bLoop = True
Switch GUIGetMsg()
Case $GUI_EVENT_CLOSE, $idCloser
ExitLoop

Case $idCheckbox
If _IsChecked($idCheckbox) Then
_Reset()
EndIf
EndSwitch

If _IsChecked($idCheckbox) Then
$iEvenOdd = _Loop($iEvenOdd)

$hStarttime = _Timer_Init()
$iTimeDiff = 0
While $iTimeDiff < $giCalc
$iTimeDiff=_Timer_Diff($hStarttime)
$array = StringSplit($iTimeDiff,'.',$STR_ENTIRESPLIT)
$left = StringLeft($array[1],1)
If Number($array[1]) < 1000 Then
$iSecValue = 0
Else
$iSecValue = $giCalc / 1000
EndIf
;~ _ArrayDisplay($array)
Switch StringLen($array[1])
Case 4
GUICtrlSetData($Label3,StringLeft($array[1],1))
Case 5
GUICtrlSetData($Label3,StringLeft($array[1],2))
Case 6
GUICtrlSetData($Label3,StringLeft($array[1],3))
EndSwitch
Sleep(250)

Switch GUIGetMsg()
Case $GUI_EVENT_CLOSE, $idCloser
$bLoop = False
ExitLoop
Case $idCheckbox
If _IsChecked($idCheckbox) Then
_Reset()
EndIf

EndSwitch
WEnd

GUICtrlSetData($Label3,0)
EndIf

WEnd

; Delete the previous GUI and all controls.
GUIDelete($hGUI)
EndFunc   ;==>Example

Func _Reset()
GUICtrlSetData($Label7,'Started: ' & _now())
$iTotal = 0
GUICtrlSetData($Label4,'Total: ' & $iTotal)
$iSkipped = 0
GUICtrlSetData($Label5,'Skips: ' & $iSkipped)
EndFunc

Func _IsChecked($idControlID)
Return BitAND(GUICtrlRead($idControlID), $GUI_CHECKED) = $GUI_CHECKED
EndFunc   ;==>_IsChecked

Func _Loop($EvenOdd)

Local $FuncName = '_Loop.' & @ScriptLineNumber

Local $aPos = MouseGetPos()
sleep(500)
Local $aPos2 = MouseGetPos()

If $aPos[0] = $aPos2[0] _
And $aPos[1] = $aPos2[1] Then
If $EvenOdd = 1 Then
$x = $aPos[0] + $giMouseAdjustment
$y = $aPos[1] + $giMouseAdjustment
Else
$x = $aPos[0] - $giMouseAdjustment
$y = $aPos[1] - $giMouseAdjustment
EndIf
GUICtrlSetData($Label1,'x: ' & $x)
GUICtrlSetData($Label2,'y: ' & $y)
MouseMove($x,$y)
$iTotal += 1
GUICtrlSetData($Label4,'Total: ' & $iTotal)
If $EvenOdd = 2 Then
$EvenOdd = 1
$sign = '+'
Else
$EvenOdd += 1
$sign = '-'
EndIf
GUICtrlSetData($Label6,$sign)
Else
$iSkipped += 1
GUICtrlSetData($Label5,'Skips: ' & $iSkipped)
EndIf

Return $EvenOdd
EndFunc