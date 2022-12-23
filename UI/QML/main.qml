import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.0
import QtGraphicalEffects 1.15
import "component"

Window {
    id: startPage
    width: 1000
    height: 750
    visible: true
    title: qsTr("Story Book Reader")
    color: "#00000000"

    // Remove title bar
    flags: Qt.Window | Qt.FramelessWindowHint
    modality: Qt.ApplicationModal

    QtObject {
        id: internal

        function switchScene() {
            var component = Qt.createComponent("app.qml")
            var win = component.createObject()
            win.show()
            visible = false
        }
    }

    Rectangle {
        id: bg
        anchors.fill: parent
        radius: 45
        clip:true

        AnimatedImage {
            id: bg_gif
            source: "../image/uibackground.gif"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            playing: true

            property bool rounded: true
            property bool adapt: true

            layer.enabled: rounded
            layer.effect: OpacityMask {
                maskSource: Item {
                    width: bg_gif.width
                    height: bg_gif.height
                    Rectangle {
                        anchors.centerIn: parent
                        width: bg_gif.adapt ? bg_gif.width : Math.min(bg_gif.width, bg_gif.height)
                        height: bg_gif.adapt ? bg_gif.height : width
                        radius: 40
                    }
                }
            }
        }

        TopBarButton{
            id: btnClose;
            visible: true
            anchors.right: parent.right;
            anchors.top: parent.top;
            enabled: true
            btnColorClicked: "#55aaff"
            btnColorMouseOver: "#ff007f"
            btnIconSource: "../icon/close_icon.svg";
            anchors.topMargin: 15
            anchors.rightMargin: 20
            onClicked: Qt.quit()
            CustomToolTip{
                text: "Close App"
            }
        }

        DropArea {
            anchors.fill: parent
            onDropped: {
                backend.get_file(drop.text)
            }
        }

        DragHandler {
            onActiveChanged: if(active){
                startPage.startSystemMove()
            }
        }
    }

    DropShadow{
        id: dropShadowBG
        opacity: 0
        anchors.fill: bg
        source: bg
        verticalOffset: 0
        horizontalOffset: 0
        radius: 10
        color: "#40000000"
        z: 1
    }

    Connections {
        target: backend

        function onSignalFile(p, l, c) {
            var component = Qt.createComponent("app.qml")
            var win = component.createObject()
            win.textFilePath = p
            win.pageNum = c
            win.lang = l
            win.show()
            visible = false
        }
    }
}


