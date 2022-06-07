<!-- # pysot-based-semi-automatic-labeling

# 1. 问题
目标检测中，对于固定视角的连续视频，标注自己的数据集显得十分困难

# 2. 方法
基于pysot的跟踪网络，我们提出一种半自动化的标注工具：

1. 使用pysot的预训练模型跟踪我们感兴趣的目标，同时在可视化窗口上显示出跟踪的结果；
2. 人工对于目标跟踪不准的情况，我们按键ESC即可中断跟踪，然后对于过去的20帧画面，使用手动标注；
3. 标注结果以YOLO的标注格式写到对应的txt文档中
 
-->
# pysot-based-semi-automatic-labeling

# 1. Questions
In target detection, it is very difficult to label your own data set for continuous videos with a fixed perspective.

# 2. Method
Based on the tracking network of pysot, we propose a semi-automatic annotation tool:

1. Use pysot's pre-trained model to track the target of interest, and display the tracking results on the visualization window;
2. If the target tracking is not accurate manually, we can press ESC to interrupt the tracking, and then use manual annotation for the past 20 frames;
3. The annotation results are written to the corresponding txt document in the YOLO annotation format

# 3. demo
