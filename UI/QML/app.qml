import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.0
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15
import "component"

Window {
    id: mainWindow
    width: 1200
    height: 720
    minimumWidth: 1100
    minimumHeight: 650
    visible: true
    color: "#00000000"
    title: qsTr("PDF")

    flags: Qt.Window | Qt.FramelessWindowHint

    property url textFilePath: "Path"
    property int pageNum
    property string lang
    property bool reading: false
    property int pageIndex: 0
    property bool enableLabel: false
    property int adaptedHeight: 0
    property int adaptedWidth: 0

    Rectangle {
        id: bg
        opacity: 1
        visible: true
        color: "#1d1d2b"
        radius: 14
        border.color: "#33334c"
        border.width: 3
        anchors.fill: parent
        clip: true
        z: 1

        Rectangle {
            id: titleBar
            height: 40
            color: "#33334c"
            radius: 14
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.rightMargin: 120
            anchors.leftMargin: 8
            anchors.topMargin: 8

            Label {
                id: labelTitleBar
                y: 14
                color: "#ffffff"
                text: qsTr("Language: " + lang)
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: iconTopLogo.right
                font.pointSize: 12
                font.family: "Segoe UI"
                anchors.leftMargin: 15
            }

            DragHandler {
                onActiveChanged: if(active){
                    mainWindow.startSystemMove()
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
            anchors.topMargin: 8
            anchors.rightMargin: 8
            onClicked: Qt.quit()
            CustomToolTip{
                text: "Close App"
            }
        }

        Rectangle {
            id: contentPages
            visible: true
            color: "#00000000"
            opacity: 1
            anchors.left: functionButtons.right
            anchors.right: parent.right
            anchors.top: titleBar.bottom
            anchors.bottom: pageTable.top
            anchors.rightMargin: 15
            anchors.leftMargin: 10
            anchors.bottomMargin: 10
            anchors.topMargin: 10
            radius: 14

            StackView {
                id: stackView
                anchors.fill: parent
                clip: true
                initialItem: image
            }

            Image {
                id: img
                visible: true
                anchors.fill: parent
                source: textFilePath+"0.jpg"
                fillMode: Image.PreserveAspectFit

                property bool rounded: true
                property bool adapt: true

                layer.enabled: rounded
                layer.effect: OpacityMask {
                    maskSource: Item {
                        width: img.width
                        height: img.height
                        Rectangle {
                            id: disparea
                            anchors.centerIn: parent
                            width: img.adapt ? img.width : Math.min(img.width, img.height)
                            height: img.adapt ? img.height : width
                        }
                    }
                }
                Component.onCompleted: backend.get_contentPage_size(height, width)
            }

            MouseArea {
                id: selectArea;
                anchors.fill: parent;
                enabled : enableLabel

                onPressed: {
                    if (highlightItem !== null) {
                        // if there is already a selection, delete it
                        highlightItem.destroy();
                    }
                    // create a new rectangle at the wanted position
                    highlightItem = highlightComponent.createObject (selectArea, {
                        "x" : mouse.x, "y" : mouse.y
                    });
                    // here you can add you zooming stuff if you want
                }

                onPositionChanged: {
                    // on move, update the width of rectangle
                    highlightItem.height = (Math.abs(mouse.y - highlightItem.y));
                    highlightItem.width = (Math.abs(mouse.x - highlightItem.x));
                }

                onReleased: {
                    enableLabel = false
                    backend.get_pos(highlightItem.x, highlightItem.y, mouse.x, mouse.y)
                    if (highlightItem !== null) {
                        highlightItem.destroy();
                    }
                    backend.label()
                }

                property Rectangle highlightItem : null;

                Component {
                    id: highlightComponent;

                    Rectangle {
                        color: "yellow"
                        opacity: 0.35
                    }
                }
            }
        }

        Flickable {
            id: pageTable
            height: 106
            contentWidth: gridLayoutBottom.width
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.rightMargin: 15
            anchors.leftMargin: 15
            anchors.bottomMargin: 4
            clip: true

            GridLayout {
                id: gridLayoutBottom
                columns: 100
                anchors.leftMargin: 0
                anchors.rightMargin: 0
                columnSpacing: 10
                rows: 0

                Repeater {
                    model: pageNum
                    CustomAppButton{
                        text: "Page " + (index+1)
                        font.pointSize: 9
                        Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                        onClicked: {
                            pageIndex = index
                            img.source = textFilePath+index+".jpg"
                            backend.get_pageIndex(index)
                        }
                    }
                }
            }

            ScrollBar.horizontal: ScrollBar {
                id: control
                size: 0.3
                position: 0.2
                orientation: Qt.Horizontal
                visible: flickable.moving || flickable.moving

                contentItem: Rectangle {
                    implicitWidth: 100
                    implicitHeight: 6
                    radius: height / 2
                    color: control.pressed ? "#55aaff" : "#40405f"
                }
            }
        }

        Column {
            id: functionButtons
            width: 50
            anchors.left: parent.left
            anchors.top: titleBar.bottom
            anchors.bottom: flickable.top
            spacing: 5
            anchors.bottomMargin: 10
            anchors.topMargin: 10
            anchors.leftMargin: 15

            CustomCircularButton {
                id: btnRead
                width: 50
                height: 50
                visible: true
                btnIconSource: "../icon/speak.svg"
                CustomToolTip {
                    text: "Read Story"
                }

                onClicked: {
                    backend.read()
                }
            }

            CustomCircularButton {
                id: btnListen
                width: 50
                height: 50
                visible: true
                btnIconSource: "../icon/listen.svg"
                CustomToolTip {
                    text: "Listen to Audio"
                }

                onClicked: {
                    backend.listen()
                }
            }

            CustomCircularButton {
                id: btnDetect
                width: 50
                height: 50
                visible: true
                btnIconSource: "../icon/detect.svg"
                CustomToolTip {
                    text: "Detect Objects"
                }

                onClicked: {
                    backend.detect()
                }
            }

            CustomCircularButton {
                id: btnLabel
                width: 50
                height: 50
                visible: true
                btnIconSource: "../icon/crop.svg"
                CustomToolTip {
                    text: "Add Label"
                }

                onClicked: {
                    backend.enable_labelling(enableLabel)
                }
            }
        }
    }

    Connections {
        target: backend



        function onSignalFinished() {
            img.source = textFilePath+"det_"+pageIndex+".jpg"
        }

        function onSignalLabelling(b) {
            enableLabel = true
        }
    }
}
