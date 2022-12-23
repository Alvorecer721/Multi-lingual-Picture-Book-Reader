## Multi-lingual Picture Book Reader

Code repository for Final year dissertation in University of Nottingham 2020-2021. 

#### Functional Prototype Demo


https://user-images.githubusercontent.com/61760910/209255009-873e469a-b0c0-45c5-9d45-e9ee6946ef45.mp4


#### Project Structure

```

14332061 MIPBR Structure
│   README.md : Project structure
│   environemtn.yml : Conda environement
│	draw_result.py: Visualisation for dissertation
│	process_image.py: Visualisation for dissertation
└───BiLSTM : 
│   │   main.py : train BiLSTM 
│   │   plot.py : plot model Performance
│   │   └───saver : tensorflow model
│   │
└───detection:
│   └───models: files borrows from github.com ultralytics yolov5
│   └───utils: files borrows from github.com ultralytics yolov5
│   │   detect.py : Detect object
│   │   best.pt: PyTorch YOLOV5 Model
│   
└───UI:
│   └───icon: icon for custom buttons
│   └───image: start scene animated background image
│   └───QML: UI Implementation
│   │   └───Components: custom buttons
│   │   │   app.qml : Main scene impementation	
│   │   │   main.qml: Start scene implementation
│   │
│   │   best.pt: PyTorch YOLOV5 Model
│   │   main.py: RUN THE PROGRAM
└───utils:
│   │   expr.py : Image Preprocessing Visualisation
│   │   listen.py : Functions for speech recognition
│   │   speech.py : Functions for speech generation
│   │   temp_folder : Temporary folder class
│   │   text.py : Functions for text extraction
```

